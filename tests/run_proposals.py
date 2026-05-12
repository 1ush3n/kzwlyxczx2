import os, sys, torch, numpy as np, pandas as pd, json
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agv_compliance_env import AGVComplianceEnv
from tsn_net.tsn_gnn_env import TSN_GNN_Env
from agent.gnn_actor_critic import GNNActorCritic
import networkx as nx

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [10, 20, 30, 42, 50]
BG_PROB = 0.3
JITTER_RANGE = (1, 10)
MAX_STEPS = 200


def load_gnn():
    gnn = GNNActorCritic(node_dim=3, edge_dim=4, hidden_dim=64).to(DEVICE)
    gnn.load_state_dict(torch.load("checkpoints/phase3_cotrain/ppo_gnn_final_aligned.pth"))
    gnn.eval()
    return gnn


def load_agv_agent():
    from stable_baselines3 import PPO
    temp_env = AGVComplianceEnv()
    return PPO.load("checkpoints/phase3_cotrain/ppo_agv_final_aligned.zip", env=temp_env, device=DEVICE)


def shortest_path_next(topo, curr, target):
    G = nx.Graph()
    for i in range(topo.num_edges):
        G.add_edge(topo.edge_index[0,i].item(), topo.edge_index[1,i].item(), weight=topo.edge_attr[i,1].item())
    try:
        return nx.dijkstra_path(G, curr, target)[1]
    except:
        return curr


def run_one(agv_agent, gnn, routing, agv_mode, proposal_cfg, seeds):
    """proposal_cfg: dict passed to AGVComplianceEnv(proposal_config=...)"""
    tsn_env = TSN_GNN_Env()
    agv_env = AGVComplianceEnv(proposal_config=proposal_cfg or {})
    fixed = np.array([0.,0.,0.], dtype=np.float32)
    m = {'rtt':[], 'stress':[], 'error':[], 'jitter':[], 'peaks':[], 'coll':0, 'de':0, 'pkt':0}
    for seed in seeds:
        rtsn=np.random.RandomState(seed); ra=np.random.RandomState(seed+1000)
        torch.manual_seed(seed)
        to,cn,am=tsn_env.reset(); agv_env.reset(seed=seed)
        for i in range(tsn_env.topo.num_edges):
            if rtsn.rand()<BG_PROB: tsn_env.gantt.check_and_add_slot(i,rtsn.uniform(0,300),rtsn.uniform(100,400))
        last_d,pk=0.,0.
        while agv_env.step_count<MAX_STEPS:
            if routing=='gnn':
                with torch.no_grad():
                    h=gnn.encode(to.to(DEVICE)); lo=gnn.get_routing_logits(h,cn,tsn_env.target_node,am.to(DEVICE))
                    nn=torch.argmax(lo).item(); ei=tsn_env._get_edge_idx(cn,nn)
                    ea=tsn_env.topo.edge_attr[ei].to(DEVICE)
                    t_offset=torch.sigmoid(gnn.scheduling_head(torch.cat([h[cn],h[nn],ea]))[0]).item()
            else:
                nn=shortest_path_next(tsn_env.topo,cn,tsn_env.target_node)
                if not am[nn]: vn=torch.where(am)[0].tolist(); nn=vn[0] if vn else cn
                t_offset=0.5
            no,ncn,nm,_,td,_,info=tsn_env.step(nn,t_offset,agv_x=agv_env.sim_engine.x_s)
            st=info.get('status','success'); m['pkt']+=1
            if st=='success': rms=(info.get('total_delay',1000)/1000.)+rtsn.uniform(*JITTER_RANGE)
            elif st=='collision': m['coll']+=1; rms=500.
            else: m['de']+=1; rms=500.
            rs=rms/1000.; jit=abs(rms-last_d); last_d=rms
            agv_env.plc.inject_tsn_delay(rs)
            ns=int(np.ceil(rs/agv_env.config['physics']['dt'])); ns=max(1,min(ns,50))
            for _ in range(ns):
                if ra.rand()<0.01: agv_env.sim_engine.x_s+=0.05; agv_env.sim_engine.v_s+=0.1
                obs=np.array(agv_env.obs_buffer,dtype=np.float32)
                if agv_mode=='rl':
                    act,_=agv_agent.predict(obs.reshape(1,*obs.shape),deterministic=True)
                    act=act.flatten()
                else:
                    act=fixed
                _,_,_,_,ai=agv_env.step(act)
                s=abs(ai['F_ext']); pk=max(pk,s); m['stress'].append(s); m['error'].append(abs(ai['error']))
            m['rtt'].append(rms); m['jitter'].append(jit)
            if td: to,cn,am=tsn_env.reset()
            for i in range(tsn_env.topo.num_edges):
                if rtsn.rand()<BG_PROB: tsn_env.gantt.check_and_add_slot(i,rtsn.uniform(0,300),rtsn.uniform(100,400))
            else: to,cn,am=no,ncn,nm
            if agv_env.step_count>=MAX_STEPS: break
        m['peaks'].append(pk)
    return m


def metrics_from_raw(m):
    def p(arr,k): return np.percentile(arr,k) if arr else 0
    return {
        'RTT': np.mean(m['rtt']), 'RTT_P95': p(m['rtt'],95), 'Jitter': np.mean(m['jitter']),
        'Coll%': m['coll']/m['pkt']*100 if m['pkt'] else 0,
        'StrM': np.mean(m['stress']), 'StrP95': p(m['stress'],95), 'StrPk': np.mean(m['peaks']),
        'Err': np.sqrt(np.mean(np.array(m['error'])**2)) if m['error'] else 0,
    }


def compare(m7_raw, m1_raw, m5_raw, label):
    m7=metrics_from_raw(m7_raw); m1=metrics_from_raw(m1_raw); m5=metrics_from_raw(m5_raw)
    print(f"\n{'='*70}")
    print(f"  M7 ({label})  vs  M1 & M5  (5 seeds)")
    print(f"{'='*70}")
    wm1=0;wm5=0
    for key,desc,n in [('RTT','RTT',1),('Jitter','Jitter',1),('Coll%','Coll%',5),('StrM','StressMean',2),('StrP95','StressP95',3),('StrPk','StressPeak',2),('Err','Error',1)]:
        g1=(m7[key]-m1[key])/abs(m1[key])*100 if m1[key] else 0
        g5=(m7[key]-m5[key])/abs(m5[key])*100 if m5[key] else 0
        w1='M7' if g1<0 else 'M1'; w5='M7' if g5<0 else 'M5'
        if g1<0: wm1+=n
        if g5<0: wm5+=n
        print(f"  {desc:15s} M7={m7[key]:10.2f} vs M1={m1[key]:10.2f} ({g1:+5.1f}%) {w1:4s} | vs M5={m5[key]:10.2f} ({g5:+5.1f}%) {w5:4s}")
    _,p1=stats.ttest_ind(m7_raw['peaks'],m1_raw['peaks'],equal_var=False)
    _,p5=stats.ttest_ind(m7_raw['peaks'],m5_raw['peaks'],equal_var=False)
    print(f"  stress_peaks t-test: vs M1 p={p1:.4f} | vs M5 p={p5:.4f}")
    max_score=sum([1,1,5,2,3,2,1])
    print(f"  SCORE: M7 beats M1 on {wm1}/{max_score} pts, beats M5 on {wm5}/{max_score} pts")
    if wm5>=12 and wm1>=10: v="STRONG ACCEPT — M7 beats both"
    elif wm5>=10: v="ACCEPT — M7 beats M5"
    elif wm5>=8: v="WEAK ACCEPT"
    else: v="REJECT"
    print(f"  VERDICT: {v}")
    return {'wm1':wm1,'wm5':wm5,'p1':p1,'p5':p5,'label':label}


# ============================================================
print("Loading models...")
gnn=load_gnn(); agv=load_agv_agent()
print("Running M1 baseline...")
m1_raw=run_one(agv,gnn,'gnn','rl',None,SEEDS)
print("Running M5 baseline...")
m5_raw=run_one(agv,gnn,'gnn','fixed',None,SEEDS)
m1m=metrics_from_raw(m1_raw); m5m=metrics_from_raw(m5_raw)
print(f"\nBASELINE: M1 RTT={m1m['RTT']:.1f} StrP95={m1m['StrP95']:.0f} | M5 RTT={m5m['RTT']:.1f} StrP95={m5m['StrP95']:.0f}")

all_results=[]

# ============================================================
#  Proposal 2: Momentum Action Smoothing
# ============================================================
for max_delta in [0.05, 0.10, 0.15, 0.20, 0.30]:
    cfg={'momentum_max_delta': max_delta}
    label=f"P2(momentum_delta={max_delta})"
    print(f"\n>>> {label}")
    raw=run_one(agv,gnn,'gnn','rl',cfg,SEEDS)
    r=compare(raw,m1_raw,m5_raw,label)
    r['label']=label; all_results.append(r)

# ============================================================
#  Proposal 1: Risk-Sensitive Reward
# ============================================================
for threshold,boost in [(2000,1.0),(1500,1.0),(1000,2.0),(800,2.0),(500,3.0)]:
    cfg={'risk_threshold': threshold, 'risk_boost': boost}
    label=f"P1(risk_thr={threshold},boost={boost})"
    print(f"\n>>> {label}")
    raw=run_one(agv,gnn,'gnn','rl',cfg,SEEDS)
    r=compare(raw,m1_raw,m5_raw,label)
    r['label']=label; all_results.append(r)

# ============================================================
#  Proposal 9: F_max override (effectively risk-sensitive)
# ============================================================
for fmax in [3000, 2000, 1500]:
    cfg={'F_max_override': fmax}
    label=f"P9(F_max={fmax})"
    print(f"\n>>> {label}")
    raw=run_one(agv,gnn,'gnn','rl',cfg,SEEDS)
    r=compare(raw,m1_raw,m5_raw,label)
    r['label']=label; all_results.append(r)

# ============================================================
#  Proposal: action smoothing (EMA blending)
# ============================================================
for sw in [0.3, 0.5, 0.7]:
    cfg={'action_smooth_weight': sw}
    label=f"P9b(action_smooth={sw})"
    print(f"\n>>> {label}")
    raw=run_one(agv,gnn,'gnn','rl',cfg,SEEDS)
    r=compare(raw,m1_raw,m5_raw,label)
    r['label']=label; all_results.append(r)

# ============================================================
#  BEST COMBO: top momentum + top risk
# ============================================================
best_combo = {'momentum_max_delta': 0.15, 'risk_threshold': 1000, 'risk_boost': 2.0}
label = "P_COMBO(mom=0.15,risk=1000,boost=2)"
print(f"\n>>> {label}")
raw=run_one(agv,gnn,'gnn','rl',best_combo,SEEDS)
r=compare(raw,m1_raw,m5_raw,label)
r['label']=label; all_results.append(r)

# ============================================================
#  RANKING
# ============================================================
print("\n"+"="*70)
print("  FINAL RANKING (weighted: StrP95 x3, Coll% x5, StrM x2)")
print("="*70)
ranked=sorted(all_results,key=lambda x: x['wm1']+x['wm5'],reverse=True)
for i,r in enumerate(ranked):
    print(f"  #{i+1}: {r['label']:45s} beat_M1={r['wm1']:2d} beat_M5={r['wm5']:2d}  p1={r['p1']:.3f} p5={r['p5']:.3f}")
