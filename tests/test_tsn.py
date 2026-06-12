import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsn_net.gantt_chart import GanttChartManager
from tsn_net.tsn_gnn_env import TSN_GNN_Env

def test_gantt_chart_circular_overlap():
    print("Testing Gantt Chart Circular Overlap...")
    gantt = GanttChartManager(num_edges=1, cycle_time=1000.0)
    
    # 插入一段跨越周期的数据: offset=0.95, duration=100 -> [950, 1000] 和 [0, 50]
    success = gantt.check_and_add_slot(0, 950.0, 100.0)
    assert success, "Initial slot should succeed."
    
    # 插入在周期开头的数据 [20, 60], 会与卷绕部分 [0, 50] 发生碰撞
    success = gantt.check_and_add_slot(0, 20.0, 40.0)
    assert not success, "Should collide with wrapped segment."
    
    # 插入在中间的数据 [500, 600], 应该成功
    success = gantt.check_and_add_slot(0, 500.0, 100.0)
    assert success, "Should succeed in middle of empty slots."
    
    print("Gantt Chart Circular Overlap test passed.\n")

def test_tsn_env_routing_and_dead_end():
    print("Testing TSN_GNN_Env Routing and Dead-End Penalty...")
    env = TSN_GNN_Env()
    obs, curr_node, mask = env.reset()
    # 固定路径测试只验证拓扑连通性，清除 reset 随机注入的背景时隙。
    env.gantt.reset()
    env._update_edge_occupancy()
    
    # 手动走几步，验证连通性
    # 拓扑：0->1->3->5->7->9
    path = [1, 3, 5, 7, 9]
    
    for next_node in path:
        assert mask[next_node].item() == True, f"Node {next_node} should be valid."
        # T_offset=0.1
        obs, curr_node, mask, reward, terminated, truncated, info = env.step(next_node, 0.1)
        
        if curr_node == 9: # Target node
            assert info['status'] == 'success', "Agent should reach target successfully."
            break
            
    print("Success Path Routing test passed.\n")
    
    # 测试死胡同逻辑
    print("Testing Dead End...")
    obs, curr_node, mask = env.reset()
    # 故意走入一个不再有可用连接的分支，假设我们走 0->2->4->6->8，如果在8处我们只能去9，但如果9也是非法的话...
    # 但由于拓扑结构，8到9是唯一的，我们可以在某处反复横跳？由于visited_nodes，不能回头。
    # 0->1->3->5->7, 如果我们在某节点无路可走
    # 为了触发 dead end，我们可以强行把所有合法邻居加入 visited_nodes
    env.visited_nodes.update(env.topo.get_neighbors(curr_node))
    # 更新 action mask
    mask = env._get_action_mask(curr_node)
    
    # 走非法步骤，期望立即触发 dead_end
    # 但必须在 step 函数里触发，由于我们手动修改了visited_nodes，再调用step传入任何动作
    # 会被防坑3检测拦截
    obs, curr_node, mask, reward, terminated, truncated, info = env.step(0, 0.5)
    
    assert terminated == True
    assert info['status'] == 'dead_end'
    assert reward == env.r_cfg['step_penalty'] + env.r_cfg['dead_end_penalty']
    print("Dead End Penalty test passed.\n")

if __name__ == "__main__":
    test_gantt_chart_circular_overlap()
    test_tsn_env_routing_and_dead_end()
    print("All TSN-GNN tests passed successfully.")
