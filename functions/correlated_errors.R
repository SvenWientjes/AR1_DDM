get_error_cor <- function(MyData){
  ERdat          <- MyData[cor==0]
  ERdat$er_speed <- -1
  ERdat[yi==1,er_speed:=ecdf(rt)(rt),by=pp]
  
  # Assign labels and shift
  ERdat$er_class <- "none"
  ERdat[stim==1 &er_speed<=0.5,er_class:="fast_1"]
  ERdat[stim==1 &er_speed >0.5,er_class:="slow_1"]
  ERdat[stim==-1&er_speed<=0.5,er_class:="fast_-1"]
  ERdat[stim==-1&er_speed >0.5,er_class:="slow_-1"]
  ERdat[,prev_er_class:=shift(er_class),by=pp]
  ERdat <- ERdat[yi==1&er_class!="none"&!is.na(prev_er_class)]
  
  # 1. Create a master template of ALL possible combinations
  all_combos <- ERdat[, CJ(pp = unique(pp), 
                           er_class = unique(er_class), 
                           prev_er_class = unique(prev_er_class))]
  
  # 2. Count your actual data
  actual_counts <- ERdat[, .(N = .N), by = .(pp, er_class, prev_er_class)]
  
  # 3. Join them together and fill the missing combinations with 0
  prop_per_pp <- actual_counts[all_combos, on = .(pp, er_class, prev_er_class)]
  prop_per_pp[is.na(N), N := 0] # Replace NA with true 0
  
  # 4. Now calculate proportions per person safely
  prop_per_pp[, prop := N / sum(N), by = .(pp, er_class)]
  prop_per_pp[is.na(prop), prop := 0]
  
  # 5. Calculate Mean, Standard Deviation, and Sample Size (number of participants)
  group_stats <- prop_per_pp[, .(
    mean_prob = mean(prop),
    sd_prob   = sd(prop),
    N_pp      = .N
  ), by = .(er_class, prev_er_class)]
  
  # 6. Calculate Standard Error (SE) and 95% Confidence Intervals
  group_stats[, se_prob := sd_prob / sqrt(N_pp)]
  group_stats[, ci_lower := mean_prob - (1.96 * se_prob)]
  group_stats[, ci_upper := mean_prob + (1.96 * se_prob)]
  group_stats[ci_lower < 0, ci_lower := 0]
  group_stats[ci_upper > 1, ci_upper := 1]
  
  return(group_stats)
}