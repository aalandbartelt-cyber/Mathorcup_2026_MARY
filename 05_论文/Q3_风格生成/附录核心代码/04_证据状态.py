"""把可比观测、未知地板和代理不稳定分开记录。"""

def classify_status(known_direct, proxy_stable, candidate_found):
    if not candidate_found:
        return '未找到搜索可行谱'
    if not known_direct:
        return '直接化学证据不足'
    if not proxy_stable:
        return '感官代理不稳定'
    return '待实测统计候选'
