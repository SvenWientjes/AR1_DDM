################################################################################
################# Replicate Vloeberghs et al. (2026) with Stan #################
################################################################################
library(cmdstanr)
library(data.table)
library(ggplot2)
library(RWiener)
source("functions/CAF.R")
################################################################################
## Load & wrangle data ----
MyData <- fread("https://raw.githubusercontent.com/kdesende/dynamic_influences_on_static_measures/refs/heads/main/data_exp2B.csv")
MyData$pp         <- ordered(MyData$sub)
levels(MyData$pp) <- c(1:99)
MyData[stim==0,stim:=-1]

# Mark trials for exclusion
MyData$yi <- 1
MyData[rt<0.1|rt>3.0,yi:=0]
################################################################################
## Visualize data ----
CAF <- calculate_group_caf(MyData[yi==1], "rt", "cor", "pp", num_bins=7)
CAF[,min_acc:=mean_acc-1.96*se_acc]
CAF[,max_acc:=mean_acc+1.96*se_acc]

## Group conditional accuracy function
ggplot(CAF, aes(x=mean_rt, y=mean_acc)) +
  geom_point() +
  geom_errorbar(aes(ymin=min_acc,ymax=max_acc),width=0.05) +
  geom_line() +
  theme_minimal()

## Example RT distribution
ggplot(MyData[pp==69&yi==1], aes(x=rt)) +
  geom_density(linetype="dashed",linewidth=1) +
  theme_minimal()

## Get temporally correlated error structure

# Get errors and speeds
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


# Interesting structure: errors both more likely to repeat speed level
# as well as response identity.
ggplot(group_stats, aes(x = er_class, y = mean_prob, fill = prev_er_class)) +
  # 1. Draw the grouped bars (dodge forces them side-by-side instead of stacked)
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), color = "black", lwd = 0.3) +
  
  # 2. Add the 95% Confidence Interval error bars
  geom_errorbar(
    aes(ymin = ci_lower, ymax = ci_upper),
    position = position_dodge(width = 0.9),
    width = 0.25, # Width of the error bar caps
    color = "grey30"
  ) +
  
  # 3. Aesthetics and Color Palette
  scale_fill_brewer(palette = "Set2", name = "Previous Error Class") +
  labs(
    title = "Transition Probabilities with 95% Confidence Intervals",
    subtitle = "Grouped by Current Error Class (Given)",
    x = "Current Error Class (er_class)",
    y = "Conditional Probability"
  ) +
  
  # 4. Clean formatting
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(size = 10),
    legend.position = "top",
    panel.grid.major.x = element_blank() # Remove vertical grids to emphasize groups
  )



################################################################################
## Fit and inspect autocorrelated DDM ----
DDMmod <- cmdstan_model("models/DDM_AR1.stan")

PNUM   <- 69
for(PNUM in 99:99){
  Pdat <- MyData[pp==PNUM]
  
  DataList <- list(N       = nrow(Pdat),
                   choice  = Pdat$response,
                   stim    = Pdat$stim,
                   y       = Pdat$rt,
                   yi      = Pdat$yi,
                   min_rt  = min(Pdat[yi==1]$rt))
  
  # Fit the model
  fit <- DDMmod$sample( #65s, 0d
    data            = DataList,
    chains          = 4,
    parallel_chains = 4,
    adapt_delta     = 0.80,
    max_treedepth   = 10,
    init_buffer     = 200,
    term_buffer     = 200,
    window          = 25,
    iter_warmup     = 1975,
    iter_sampling   = 2000,
    output_dir      = "fits/AR1",
    output_basename = paste0("AR1_pp",formatC(PNUM,width=2,flag="0"))
  )
}

TheSum <- data.table(fit$summary())
any(TheSum$rhat>1.01)

# Plot parameter over time
v_draws  <- fit$draws("v",format="matrix")
w_draws  <- fit$draws("w",format="matrix")
a_draws  <- fit$draws("a",format="matrix")
t0_draws <- fit$draws("t0",format="matrix")

# Plot parameter time-course
ggplot() +
  geom_line(aes(x=1:dim(v_draws)[2], y=apply(v_draws,2,mean))) +
  geom_smooth(aes(x=1:dim(v_draws)[2], y=apply(v_draws,2,mean)))#+
  #geom_ribbon(aes(x=1:474, ymin=apply(w_draws,2,function(x){bayestestR::hdi(x)$CI_low}),
  #                         ymax=apply(w_draws,2,function(x){bayestestR::hdi(x)$CI_high})),
  #            alpha=0.2)



total_sims <- 500 * dim(v_draws)[2] # replace 500 everywhere with dim(v_draws)[1]
sim_rt     <- numeric(total_sims)
sim_choice <- character(total_sims)
si <- 1
for(s in 1:500){
  print(paste0("Simulating sample ",s,"/",dim(v_draws)[1]))
  for(t in 1:dim(v_draws)[2]){
    v_trial <- Pdat[t,stim] * v_draws[s,t]
    simed   <- rwiener(n=1,
                       alpha = a_draws[s,t],
                       tau   = t0_draws[s],
                       beta  = w_draws[s,t],
                       delta = v_trial)
    sim_rt[si]     <- simed$q
    sim_choice[si] <- simed$resp
    si             <- si+1
  }
}
SIM <- data.table(
  s     = rep(1:500, each = dim(v_draws)[2]),
  t     = rep(1:dim(v_draws)[2], 500),
  rt     = sim_rt,
  choice = sim_choice
)

ggplot() +
  geom_density(data=MyData[pp==PNUM&yi==1], aes(x=rt), linetype="dashed",linewidth=1) +
  geom_density(data=SIM[rt>0.1&rt<3.0],aes(x=rt))

################################################################################
## Check Rhat and ESS for fits ----
for(PNUM in 14:99){
  DDMfit <- as_cmdstan_fit(paste0("fits/AR1/AR1_pp",formatC(PNUM,width=2,flag="0"),"-",1:4,".csv"))
  TheSum <- DDMfit$summary()
  print(paste0("---- PP",formatC(PNUM,width=2,flag="0")," ----"))
  print("BAD RHAT:")
  print(which(TheSum$rhat > 1.01))
  print("BAD ESS BULK:")
  print(which(TheSum$ess_bulk < 1000))
  print("BAD ESS TAIL:")
  print(which(TheSum$ess_tail < 1000))
}

# Sample from every participant
N_sims       <- 500
total_sims   <- N_sims * 500
SIM_list     <- list()
downsamp_idx <- round(seq(1,8000,length.out=N_sims))
for(PNUM in 1:99){
  print(paste0("=================================="))
  print(paste0("Simulating participant ",PNUM,"/99"))
  
  # Get participant data and fit
  Pdat   <- MyData[pp==PNUM]
  DDMfit <- as_cmdstan_fit(paste0("fits/AR1/AR1_pp",formatC(PNUM,width=2,flag="0"),"-",1:4,".csv"))
  
  # Get parameters over time
  v_draws  <- DDMfit$draws("v",format="matrix")[downsamp_idx,]
  w_draws  <- DDMfit$draws("w",format="matrix")[downsamp_idx,]
  a_draws  <- DDMfit$draws("a",format="matrix")[downsamp_idx,]
  t0_draws <- DDMfit$draws("t0",format="matrix")[downsamp_idx,]
  
  # Get simulated choice/RTs
  sim_rt     <- numeric(total_sims)
  sim_choice <- character(total_sims)
  was_dir    <- character(total_sims)
  
  si <- 1
  for(s in 1:N_sims){
    print(paste0("    Simulating sample ",s,"/",dim(v_draws)[1]))
    for(t in 1:dim(v_draws)[2]){
      v_trial <- Pdat[t,stim] * v_draws[s,t]
      simed   <- rwiener(n=1,
                         alpha = a_draws[s,t],
                         tau   = t0_draws[s],
                         beta  = w_draws[s,t],
                         delta = v_trial)
      sim_rt[si]     <- simed$q
      sim_choice[si] <- as.character(simed$resp)
      was_dir[si]    <- if(Pdat[t,stim]>0){"upper"}else{"lower"}
      si             <- si+1
    }
  }
  SIM_list[[PNUM]] <- data.table(
    p     = PNUM,
    sim   = rep(1:N_sims, each = dim(v_draws)[2]),
    t     = rep(1:dim(v_draws)[2], N_sims),
    rt     = sim_rt,
    choice = sim_choice,
    was_dir = was_dir,
    correct = sim_choice==was_dir
  )
}
SIM <- rbindlist(SIM_list)
fwrite(SIM, "simulations/simulations.csv.gz", compress = "gzip", compressLevel = 9)



thesim <- fread("simulations/simulations.csv.gz")


plot(density(SIM[p==69&rt>0.1&rt<3.0]$rt))


Pdat <- MyData[pp==15&yi==1]
Psim <- SIM[p==15&rt>0.1&rt<3.0]

ggplot() +
  geom_density(data=Pdat,aes(x=rt),linetype="dashed",linewidth=1) +
  geom_density(data=Psim,aes(x=rt),linetype="dotdash",linewidth=1,color="orange") +
  theme_minimal()



## Simulated CAF
SIM[,stim:=rep(MyData$stim,N_sims)]
SIM$cor <- 0
SIM[stim==1&choice==1,cor:=1]
SIM[stim==-1&choice==2,cor:=1]

CAFlist <- list()
for(simje in 1:SIM[,max(sim)]){
  
  # Assign stimulus identity and correctness
  insim <- SIM[sim==simje]
  insim[,stim:=MyData$stim]
  insim$cor <- 0
  insim[stim==1&choice==1,cor:=1]
  insim[stim==-1&choice==2,cor:=1]
  
  CAFlist[[simje]] <- calculate_group_caf(SIM[sim==simje], "rt", "cor", "p", num_bins=7)
}
CAF_sim <- rbindlist(CAFlist)


calculate_group_caf(SIM[sim==1], "rt", "cor", "p", num_bins=7)


