import numpy as np
from typing import List, Tuple

class GanttChartManager:
    """
    TSN 网络边调度甘特图管理器
    支持基于 GCL (Gate Control List) 周期 (T_cycle) 的卷绕碰撞检测 (Circular Overlap Detection)。
    """
    def __init__(self, num_edges: int, cycle_time: float):
        self.num_edges = num_edges
        self.cycle_time = cycle_time
        
        # 每个边维护一个已占用时间槽的列表 [(start1, end1), (start2, end2), ...]
        self.edge_slots = [[] for _ in range(num_edges)]
        
    def reset(self):
        """清空所有边的时间槽"""
        self.edge_slots = [[] for _ in range(self.num_edges)]
        
    def _check_overlap_1d(self, s1: float, e1: float, s2: float, e2: float) -> bool:
        """检查两个普通的一维线段是否重叠"""
        return max(s1, s2) < min(e1, e2)
        
    def check_and_add_slot(self, edge_idx: int, start_time: float, duration: float) -> bool:
        """
        在特定的边上分配一个时间窗，包含周期卷绕的检查。
        :param edge_idx: 边索引
        :param start_time: 绝对开始时间 (不受周期限制，可以是很大的值)
        :param duration: 传输持续时间
        :return: 如果成功分配（无碰撞）返回 True；如果发生碰撞返回 False。
        """
        assert duration <= self.cycle_time, "Duration cannot exceed one full cycle."
        
        # 将 start_time 映射到单个周期内 [0, T_cycle)
        mod_start = start_time % self.cycle_time
        mod_end = mod_start + duration
        
        # 处理可能的卷绕 (Wrap-around)
        # 如果 mod_end > T_cycle，说明这个块跨越了周期边界，分为两段：
        # 段1: [mod_start, T_cycle]
        # 段2: [0, mod_end - T_cycle]
        segments_to_check = []
        if mod_end > self.cycle_time:
            segments_to_check.append((mod_start, self.cycle_time))
            segments_to_check.append((0.0, mod_end - self.cycle_time))
        else:
            segments_to_check.append((mod_start, mod_end))
            
        existing_slots = self.edge_slots[edge_idx]
        
        # 碰撞检测
        for (es, ee) in existing_slots:
            for (cs, ce) in segments_to_check:
                if self._check_overlap_1d(cs, ce, es, ee):
                    return False # 发生重叠冲突
                    
        # 若无冲突，将新块持久化加入（同样以分离后的段形式加入）
        for (cs, ce) in segments_to_check:
            existing_slots.append((cs, ce))
            
        return True
