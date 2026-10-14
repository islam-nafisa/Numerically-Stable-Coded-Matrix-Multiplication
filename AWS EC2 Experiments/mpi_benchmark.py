"""Memory-aware, large-payload MPI benchmark for coded matrix multiplication.

Rank 0: master; ranks 1..N: workers. Two groups share the FP64 baseline:
  decimal quantization + lossless bit packing: fp64, dec6, dec5, dec4, dec3
  fixed-width IEEE floating transmission: fp64, fp32, fp16
Each mode is repeated for trials independent input matrix pairs (same input per
trial across modes). Outputs: per_trial.csv, aggregate.csv, wire_formats.csv.

Timing definitions:
  down: immediately before metadata + chunked A/B sends, ends after all ACKs
  up: immediately before GO commands for first tau READY ranks, ends when
      all tau result payloads received; READY and compute/packing excluded.

This implementation deliberately buffers all tau worker return payloads on
master to EXCLUDE decoding/unpacking from network timing. Requires significant
RAM for 30k size (about 166 GB *just* for FP64 return buffers).
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import math
from pathlib import Path

import numpy as np
from mpi4py import MPI
from threadpoolctl import threadpool_limits

from coefficients import load_design
from wire_codec import pack_array, unpack_array, unpack_chunks, MODES, verify_descriptor

TAG_PREP=10
TAG_A=11
TAG_B=12
TAG_ACK=13
TAG_COMPUTE_GO=14
TAG_READY=15
TAG_RETURN_GO=16
TAG_RESULT=17
TAG_WARM=18

GROUPS = {
    'packed': ['fp64','dec6','dec5','dec4','dec3'],
    'fixed': ['fp64','fp32','fp16'],
    'both': list(MODES),
}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ours', type=Path, required=True, help='Optimized coefficient MAT file on master')
    p.add_argument('--n',type=int,default=25)
    p.add_argument('--p',type=int,default=12)
    p.add_argument('--tau',type=int,default=23)
    p.add_argument('--size',type=int,default=30000)
    p.add_argument('--trials',type=int,default=5)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--groups',choices=GROUPS,default='both')
    p.add_argument('--output',type=Path,default=Path('results/full_30k'))
    p.add_argument('--threads',type=int,default=8,help='BLAS threads per worker')
    p.add_argument('--master-threads',type=int,default=32,help='BLAS threads for master reference')
    p.add_argument('--chunk-mib',type=int,default=16,help='Maximum MPI numeric message MiB')
    p.add_argument('--index-base',choices=('auto','zero','one'),default='auto')
    p.add_argument('--resume',action='store_true',help='Skip completed (trial,mode) records in existing CSV')
    p.add_argument('--confirm-large-run',action='store_true',help='Required to run matrices >=10000')
    return p.parse_args()


def append_csv(path, rows):
    if not rows:
        return
    with path.open('a', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


def load_csv(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def summarize(output_dir):
    """Recompute aggregate from saved per-trial rows after each completed mode."""
    rows=load_csv(output_dir/'per_trial.csv')
    result=[]
    for mode in MODES:
        group=[r for r in rows if r['mode']==mode]
        if not group:
            continue
        def fld(name): return np.array([float(r[name]) for r in group],dtype=np.float64)
        result.append({
            'experiment_group': 'shared' if mode=='fp64' else ('bitpacked' if mode.startswith('dec') else 'fixed_float'),
            'mode':mode,'completed_trials':len(group),
            'mean_master_to_workers_ms':float(fld('master_to_workers_s').mean()*1000),
            'mean_workers_to_master_ms':float(fld('workers_to_master_s').mean()*1000),
            'mean_master_to_workers_payload_MB':float(fld('master_to_workers_payload_MB').mean()),
            'mean_workers_to_master_payload_MB':float(fld('workers_to_master_payload_MB').mean()),
            'mean_relative_error':float(fld('relative_error').mean()),
            'max_relative_error':float(fld('relative_error').max()),
            'max_master_to_workers_ms':float(fld('master_to_workers_s').max()*1000),
            'max_workers_to_master_ms':float(fld('workers_to_master_s').max()*1000),
        })
    path=output_dir/'aggregate.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        if result:
            writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)


def mpi_send_bytes(comm, payload, dest, tag, chunk):
    if payload.dtype != np.uint8 or not payload.flags.c_contiguous:
        raise TypeError('Wire payload must be contiguous uint8')
    for lo in range(0,len(payload),chunk):
        comm.Send([payload[lo:lo+chunk],MPI.BYTE],dest=dest,tag=tag)


def mpi_recv_bytes(comm, payload, source, tag, chunk):
    for lo in range(0,len(payload),chunk):
        comm.Recv([payload[lo:lo+chunk],MPI.BYTE],source=source,tag=tag)


def worker(comm, rank, args, chunk):
    expected_input_shape=(args.size//args.p,args.size)
    while True:
        instructions=comm.recv(source=0,tag=TAG_PREP)
        if instructions.get('stop'):
            return
        if instructions.get('warmup'):
            warm=np.empty(1024*1024,dtype=np.uint8)
            comm.Recv([warm,MPI.BYTE],source=0,tag=TAG_WARM)
            comm.send(True,dest=0,tag=TAG_ACK)
            comm.recv(source=0,tag=TAG_COMPUTE_GO)
            with threadpool_limits(limits=args.threads):
                _=np.ones((64,64),dtype=np.float64)@np.ones((64,64),dtype=np.float64)
            comm.Send([warm,MPI.BYTE],dest=0,tag=TAG_WARM)
            continue

        desc_a=instructions['a']
        desc_b=instructions['b']
        if tuple(desc_a['shape'])!=expected_input_shape or tuple(desc_b['shape'])!=expected_input_shape:
            raise ValueError(f'Incorrect received matrix shape; expected {expected_input_shape}')
        verify_descriptor(desc_a)
        verify_descriptor(desc_b)
        pa=np.empty(int(desc_a['nbytes']),dtype=np.uint8)
        pb=np.empty(int(desc_b['nbytes']),dtype=np.uint8)
        mpi_recv_bytes(comm,pa,0,TAG_A,chunk)
        mpi_recv_bytes(comm,pb,0,TAG_B,chunk)
        comm.send(True,dest=0,tag=TAG_ACK)

        comm.recv(source=0,tag=TAG_COMPUTE_GO)
        a=unpack_array(pa,desc_a)
        b=unpack_array(pb,desc_b)
        del pa,pb
        with threadpool_limits(limits=args.threads):
            result=a.T@b
        del a,b
        packed_result,result_desc=pack_array(result,instructions['mode'])
        del result
        comm.send({'result':result_desc},dest=0,tag=TAG_READY)
        action=comm.recv(source=0,tag=TAG_RETURN_GO)
        if action=='send':
            mpi_send_bytes(comm,packed_result,0,TAG_RESULT,chunk)
        elif action!='skip':
            raise ValueError(f'Unexpected return instruction {action!r}')
        del packed_result


def warmup(comm,args):
    """1 MiB transport in both directions and small BLAS call, outside both timers."""
    block=np.zeros(1024*1024,dtype=np.uint8)
    for wid in range(args.n):
        comm.send({'warmup':True},dest=wid+1,tag=TAG_PREP)
        comm.Send([block,MPI.BYTE],dest=wid+1,tag=TAG_WARM)
    for wid in range(args.n):
        comm.recv(source=wid+1,tag=TAG_ACK)
    for wid in range(args.n):
        comm.send(True,dest=wid+1,tag=TAG_COMPUTE_GO)
    for wid in range(args.n):
        comm.Recv([block,MPI.BYTE],source=wid+1,tag=TAG_WARM)


def generate_input(size,p,seed,master_threads):
    rng=np.random.default_rng(seed)
    a=np.empty((size,size),dtype=np.float64)
    b=np.empty((size,size),dtype=np.float64)
    rng.standard_normal(size=(size,size),dtype=np.float64,out=a)
    rng.standard_normal(size=(size,size),dtype=np.float64,out=b)
    a /= math.sqrt(size)
    b /= math.sqrt(size)
    print(f'Input A/B generated: seed={seed}',flush=True)
    with threadpool_limits(limits=master_threads):
        reference=a.T@b
    print('Full-precision reference matrix computed',flush=True)
    return a.reshape(p,size//p,size),b.reshape(p,size//p,size),reference


def squared_norm(arr, block=1_000_000):
    flat=arr.reshape(-1)
    accum=0.0
    for lo in range(0,len(flat),block):
        x=flat[lo:lo+block]
        accum+=float(np.dot(x,x))
    return accum


def relative_error(reconstruction,reference,ref_squared,block=1_000_000):
    a=reconstruction.reshape(-1)
    b=reference.reshape(-1)
    squared=0.0
    for lo in range(0,len(a),block):
        d=a[lo:lo+block]-b[lo:lo+block]
        squared+=float(np.dot(d,d))
    return float(math.sqrt(squared/ref_squared))


def select_ready(comm,tau,n):
    selected=[];reports={}
    while len(selected)<tau:
        status=MPI.Status()
        report=comm.recv(source=MPI.ANY_SOURCE,tag=TAG_READY,status=status)
        wid=status.Get_source()-1
        if wid<0 or wid>=n or wid in reports:
            raise RuntimeError(f'Invalid/duplicate worker READY id {wid}')
        selected.append(wid)
        reports[wid]=report
    return selected,reports


def receive_selected_timed(comm, selected,reports,chunk):
    """Post first chunk before start, GO selected, then keep 1 Irecv per worker.

    All returned buffers are kept in RAM. The timer stops BEFORE unpacking,
    decoding, or freeing these buffers. Each chunk < MPI legacy 32-bit count.
    """
    buffers={}
    reqs=[]
    transferred=[]
    for wid in selected:
        desc=reports[wid]['result']
        verify_descriptor(desc)
        size=int(desc['nbytes'])
        buf=np.empty(size,dtype=np.uint8)
        buffers[wid]=buf
        first=min(size,chunk)
        reqs.append(comm.Irecv([buf[:first],MPI.BYTE],source=wid+1,tag=TAG_RESULT))
        transferred.append(first)
    start=MPI.Wtime()
    for wid in selected:
        comm.send('send',dest=wid+1,tag=TAG_RETURN_GO)
    while True:
        index=MPI.Request.Waitany(reqs)
        if index==MPI.UNDEFINED:
            break
        wid=selected[index]
        pos=transferred[index]
        if pos==buffers[wid].nbytes:
            reqs[index]=MPI.REQUEST_NULL
        else:
            end=min(pos+chunk,buffers[wid].nbytes)
            reqs[index]=comm.Irecv([buffers[wid][pos:end],MPI.BYTE],source=wid+1,tag=TAG_RESULT)
            transferred[index]=end
    elapsed=MPI.Wtime()-start
    return buffers, elapsed


def skip_stragglers(comm, n, selected):
    leftover=n-len(selected)
    skipped=[]
    while len(skipped)<leftover:
        status=MPI.Status()
        comm.recv(source=MPI.ANY_SOURCE,tag=TAG_READY,status=status)
        wid=status.Get_source()-1
        if wid in selected or wid in skipped or wid<0 or wid>=n:
            raise RuntimeError(f'Unexpected worker READY from {wid}')
        skipped.append(wid)
        comm.send('skip',dest=wid+1,tag=TAG_RETURN_GO)
    return skipped


def run_master(comm,args,chunk):
    print('Loading optimized coefficients...',flush=True)
    design=load_design(args.ours,args.n,args.p,args.tau,args.index_base)
    args.output.mkdir(parents=True,exist_ok=True)
    prior=load_csv(args.output/'per_trial.csv')
    if prior and not args.resume:
        raise FileExistsError('Existing per_trial.csv: specify a NEW output directory or --resume')
    modes=GROUPS[args.groups]
    finished=set()
    for row in prior:
        if (int(row['N']),int(row['p']),int(row['tau']),int(row['size']),int(row['seed_base'])) != (args.n,args.p,args.tau,args.size,args.seed):
            raise ValueError('Existing results have different settings; choose a different output folder')
        finished.add((int(row['trial']),row['mode']))
    print('Unmeasured small-message MPI/BLAS warm-up...',flush=True)
    warmup(comm,args)
    print('Warm-up complete',flush=True)

    for trial in range(1,args.trials+1):
        pending=[mode for mode in modes if (trial,mode) not in finished]
        if not pending:
            print(f'Skipping complete trial {trial}',flush=True)
            continue
        print(f'\n===== INPUT TRIAL {trial}/{args.trials} =====',flush=True)
        a_blocks,b_blocks,reference=generate_input(args.size,args.p,args.seed+trial-1,args.master_threads)
        ref_squared=squared_norm(reference)
        if ref_squared<=0: raise ValueError('Reference has zero norm')
        for mode in pending:
            print(f'\nTrial {trial}/{args.trials}, mode={mode}: encoding/packing inputs (not timed)...',flush=True)
            prepared={}
            for wid in range(args.n):
                ae=np.einsum('p,pij->ij',design.x[wid],a_blocks,optimize=True)
                be=np.einsum('p,pij->ij',design.y[wid],b_blocks,optimize=True)
                pa,da=pack_array(ae,mode)
                pb,db=pack_array(be,mode)
                prepared[wid]=(pa,da,pb,db)
                del ae,be
            print(f'  All {args.n} pairs ready. Starting master-to-workers clock.',flush=True)

            # FIRST timer: sequential input transmissions, wait for all ACKs.
            distribution_start=MPI.Wtime()
            for wid in range(args.n):
                pa,da,pb,db=prepared[wid]
                comm.send({'mode':mode,'a':da,'b':db},dest=wid+1,tag=TAG_PREP)
                mpi_send_bytes(comm,pa,wid+1,TAG_A,chunk)
                mpi_send_bytes(comm,pb,wid+1,TAG_B,chunk)
            for wid in range(args.n):
                comm.recv(source=wid+1,tag=TAG_ACK)
            dist_s=MPI.Wtime()-distribution_start
            outbound=sum(info[1]['nbytes']+info[3]['nbytes'] for info in prepared.values())
            input_desc={wid:(info[1],info[3]) for wid,info in prepared.items()}
            del prepared
            print(f'  Distribution finished: {dist_s:.2f}s. Releasing workers to compute...',flush=True)

            for wid in range(args.n):
                comm.send(True,dest=wid+1,tag=TAG_COMPUTE_GO)
            selected,reports=select_ready(comm,args.tau,args.n)
            print(f'  First {args.tau} ready workers: {selected}. Starting return clock...',flush=True)
            for wid in selected:
                desc=reports[wid]['result']
                verify_descriptor(desc)
                if tuple(desc['shape'])!=(args.size,args.size):
                    raise ValueError('Result matrix has wrong dimensions')
            # SECOND timer: GO -> full payload arrival; decoding is NOT timed.
            buffers,return_s=receive_selected_timed(comm,selected,reports,chunk)
            print(f'  Return finished: {return_s:.2f}s. Unpacking/decoding (not timed)...',flush=True)
            skipped=skip_stragglers(comm,args.n,selected)

            alpha=design.decoder(selected)
            reconstruction=np.zeros((args.size,args.size),dtype=np.float64)
            reconstructed=reconstruction.reshape(-1)
            for wid in selected:
                for offset,values in unpack_chunks(buffers[wid],reports[wid]['result']):
                    reconstructed[offset:offset+len(values)]+=float(alpha[wid])*values
                del buffers[wid]  # release each large payload after it has been decoded
            del buffers
            err=relative_error(reconstruction,reference,ref_squared)
            del reconstruction,reconstructed

            identity=np.zeros((args.p,args.p),dtype=np.float64)
            for wid,weight in alpha.items():
                identity+=weight*np.outer(design.x[wid],design.y[wid])
            coeff_err=float(np.linalg.norm(identity-np.eye(args.p))/math.sqrt(args.p))
            inbound=sum(reports[wid]['result']['nbytes'] for wid in selected)
            record={
                'trial':trial,'seed_base':args.seed,'input_seed':args.seed+trial-1,
                'mode':mode,'N':args.n,'p':args.p,'tau':args.tau,'size':args.size,
                'selected_workers':';'.join(map(str,selected)),
                'skipped_workers':';'.join(map(str,skipped)),
                'master_to_workers_s':dist_s,'workers_to_master_s':return_s,
                'master_to_workers_payload_bytes':outbound,
                'workers_to_master_payload_bytes':inbound,
                'master_to_workers_payload_MB':outbound/1e6,
                'workers_to_master_payload_MB':inbound/1e6,
                'relative_error':err,'coefficient_residual':coeff_err,
            }
            wires=[]
            for wid in range(args.n):
                da,db=input_desc[wid]
                tuples=[('master_to_worker','A',da),('master_to_worker','B',db)]
                if wid in reports:
                    tuples.append(('worker_to_master','result',reports[wid]['result']))
                for direction,name,desc in tuples:
                    wires.append({
                        'trial':trial,'mode':mode,'worker_id':wid,
                        'selected_for_return':wid in reports,
                        'direction':direction,'matrix':name,
                        'shape':'x'.join(map(str,desc['shape'])),
                        'packing':desc['packing'],
                        'bits_per_value':desc['bits_per_value'],
                        'decimal_places':desc['decimal_places'],
                        'payload_bytes':desc['nbytes']
                    })
            append_csv(args.output/'wire_formats.csv',wires)
            append_csv(args.output/'per_trial.csv',[record])
            summarize(args.output)
            print(f"DONE trial {trial} {mode}: down={dist_s:.3f}s/{outbound/1e6:.3f}MB "
                  f"up={return_s:.3f}s/{inbound/1e6:.3f}MB error={err:.6e}",flush=True)
            gc.collect()
        del a_blocks,b_blocks,reference
        gc.collect()
    for rank in range(1,args.n+1):
        comm.send({'stop':True},dest=rank,tag=TAG_PREP)
    print(f'ALL REQUESTED MODES COMPLETE. Results at {args.output}',flush=True)


def main():
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    args=parse_args()
    if args.size>=10000 and not args.confirm_large_run:
        raise SystemExit('SAFETY: --confirm-large-run is required for matrices >= 10000')
    if not (1<=args.p<=args.tau<=args.n) or args.size%args.p or args.trials<1:
        raise SystemExit('Require 1<=p<=tau<=n and matrix size divisible by p, trials>=1')
    if args.threads<1 or args.master_threads<1 or not (1<=args.chunk_mib<=128):
        raise SystemExit('Thread counts and chunk-MiB must be in permitted ranges')
    if comm.Get_size()!=args.n+1:
        raise SystemExit(f'Expected exactly {args.n+1} MPI ranks; got {comm.Get_size()}')
    chunk=args.chunk_mib*1024*1024
    try:
        if rank==0:
            run_master(comm,args,chunk)
        else:
            worker(comm,rank,args,chunk)
    except Exception as exc:
        print(f'FATAL MPI rank {rank}: {type(exc).__name__}: {exc}',flush=True)
        comm.Abort(1)
        raise

if __name__=='__main__':
    main()
