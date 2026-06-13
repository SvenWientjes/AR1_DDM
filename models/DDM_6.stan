data {
  int<lower=1> N;                        // Number of trials
  array[N] int<lower=-1,upper=1> stim;   // stimulus (1 or -1)
  array[N] int<lower=0,upper=1>  choice; // Upper (1) or lower (0) boundary
  array[N] int<lower=0,upper=1>  yi;     // Trial inclusion (0:no, 1:yes)
  vector[N] y;                           // RTs (s)
  real<lower=0> min_rt;                  // Minimum observed RT (to constrain t0)
}

parameters {
  // Parameterization and priors as in https://doi.org/10.3758/s13428-023-02179-1
  real<lower=0>          a;
  real                   v;
  real<lower=0, upper=1> w;
  real<lower=0>          t0;
  real<lower=0>          sv;
  real<lower=0, upper=1> sw;
}
model {
  // Priors (vaguely informative)
  a ~ normal(1,1);
  w ~ normal(0.5,0.1);
  v ~ normal(2,3);
  t0 ~ normal(0.435,0.12);
  sv ~ normal(1,3);
  sw ~ beta(1,3);
  

  for (n in 1:N) {
    if(yi[n]==1){
      real v_trial = stim[n]*v;
      if(choice[n] == 1){
        y[n] ~ wiener(a, t0, w, v_trial,sv,sw,0.0);
      }else{
        y[n] ~ wiener(a, t0, 1 - w, -v_trial,sv,sw,0.0);
      }
    }
  }
}
