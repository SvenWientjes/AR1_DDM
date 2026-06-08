calculate_group_caf <- function(data, rt_col, acc_col, id_col, num_bins = 5) {
  # Convert to data.table if it isn't one already without copying
  dt <- as.data.table(data)
  
  # Step 1 & 2: Calculate bins and aggregate at the individual participant level
  # frank(..., ties.method = "first") creates the ranking needed for quantiles
  ind_dt <- dt[, {
    # Generate bin assignments based on individual RT quantiles
    rt_bin <- as.integer(cut(frank(get(rt_col), ties.method = "first"), 
                             breaks = num_bins, 
                             labels = FALSE))
    
    # Aggregate within each bin for this specific participant
    .(rt_bin       = rt_bin, 
      ind_mean_rt  = get(rt_col), 
      ind_mean_acc = get(acc_col))
  }, by = c(id_col)][, .(
    ind_mean_rt  = mean(ind_mean_rt, na.rm = TRUE),
    ind_mean_acc = mean(ind_mean_acc, na.rm = TRUE)
  ), by = c(id_col, "rt_bin")]
  
  # Step 3: Aggregate across participants to get the group-level CAF
  group_caf <- ind_dt[, .(
    mean_rt  = mean(ind_mean_rt, na.rm = TRUE),
    mean_acc = mean(ind_mean_acc, na.rm = TRUE),
    se_acc   = sd(ind_mean_acc, na.rm = TRUE) / sqrt(.N)
  ), by = .(rt_bin)][order(rt_bin)] # Ensure bins are sorted 1 to N
  
  return(group_caf)
}