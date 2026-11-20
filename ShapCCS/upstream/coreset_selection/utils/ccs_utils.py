import torch


def mislabel_mask(data_score, mis_num, mis_descending):
    mis_score = data_score
    mis_score_sorted_index = mis_score.argsort()
    if mis_descending:
        mis_score_sorted_index = mis_score_sorted_index[::-1]
    easy_index = mis_score_sorted_index[mis_num:]
    data_score = data_score[easy_index]

    return data_score, easy_index

def stratified_sampling(data_score, coreset_num):
    stratas = 50
    # print('Using stratified sampling...')
    score = data_score
    total_num = coreset_num
    score = torch.tensor(score)
    min_score = torch.min(score)
    max_score = torch.max(score) * 1.0001
    step = (max_score - min_score) / stratas

    def bin_range(k):
        return min_score + k * step, min_score + (k + 1) * step

    strata_num = []
    ##### calculate number for each strata #####
    for i in range(stratas):
        start, end = bin_range(i)
        num = torch.logical_and(score >= start, score < end).sum()
        strata_num.append(num)

    strata_num = torch.tensor(strata_num)

    def bin_allocate(num, bins):
        sorted_index = torch.argsort(bins)
        sort_bins = bins[sorted_index]

        num_bin = bins.shape[0]

        rest_exp_num = num
        budgets = []
        for i in range(num_bin):
            rest_bins = num_bin - i
            avg = rest_exp_num // rest_bins
            cur_num = min(sort_bins[i].item(), avg)
            budgets.append(cur_num)
            rest_exp_num -= cur_num

        rst = torch.zeros((num_bin,)).type(torch.int)
        rst[sorted_index] = torch.tensor(budgets).type(torch.int)

        return rst

    budgets = bin_allocate(total_num, strata_num)

    ##### sampling in each strata #####
    selected_index = []
    sample_index = torch.arange(data_score.shape[0])

    for i in range(stratas):
        start, end = bin_range(i)
        mask = torch.logical_and(score >= start, score < end)
        pool = sample_index[mask]
        rand_index = torch.randperm(pool.shape[0])
        selected_index += [idx.item() for idx in pool[rand_index][:budgets[i]]]

    return selected_index, None
