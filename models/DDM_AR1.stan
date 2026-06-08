data {
  int<lower=1> N;                        // Number of trials
  array[N] int<lower=-1,upper=1> stim;   // stimulus (1 or -1)
  array[N] int<lower=0,upper=1>  choice; // Upper (1) or lower (0) boundary
  array[N] int<lower=0,upper=1>  yi;     // Trial inclusion (0:no, 1:yes)
  vector[N] y;                           // RTs (s)
  real<lower=0> min_rt;                  // Minimum observed RT (to constrain t0)
}

parameters {
  // Mean levels
  real mu_log_a;
  real mu_logit_w;
  real mu_v;

  // AR(1) persistence
  real rho_a_raw;
  real rho_w_raw;
  real rho_v_raw;

  // Innovation scales
  real log_sigma_a;
  real log_sigma_w;
  real log_sigma_v;

  // Non-centered standard normal innovations
  vector[N] a_z;
  vector[N] w_z;
  vector[N] v_z;

  // Non-decision time
  real t0_raw;
}

transformed parameters {
  real rho_a = inv_logit(rho_a_raw);
  real rho_w = inv_logit(rho_w_raw);
  real rho_v = inv_logit(rho_v_raw);
  
  vector[N] a_dev;
  vector[N] w_dev;
  vector[N] v_dev;

  vector[N] a;
  vector[N] w;
  vector[N] v;

  real<lower=0> t0;
  
  real<lower=0> sigma_a = exp(log_sigma_a);
  real<lower=0> sigma_w = exp(log_sigma_w);
  real<lower=0> sigma_v = exp(log_sigma_w);

  t0 = Phi(t0_raw) * min_rt;

  // Unit-variance AR(1) states
  a_dev[1] = a_z[1];
  w_dev[1] = w_z[1];
  v_dev[1] = v_z[1];

  for (n in 2:N) {
    a_dev[n] = rho_a * a_dev[n - 1] + sqrt(1 - square(rho_a)) * a_z[n];
    w_dev[n] = rho_w * w_dev[n - 1] + sqrt(1 - square(rho_w)) * w_z[n];
    v_dev[n] = rho_v * v_dev[n - 1] + sqrt(1 - square(rho_v)) * v_z[n];
  }

  // Map latent states to DDM parameter scales
  for (n in 1:N) {
    a[n] = exp(mu_log_a + sigma_a * a_dev[n]);
    w[n] = Phi(mu_logit_w + sigma_w * w_dev[n]);
    v[n] = mu_v + sigma_v * v_dev[n];
  }
}

model {
  // Priors on means
  mu_log_a   ~ normal(0, 0.5);
  mu_logit_w ~ std_normal();
  mu_v       ~ normal(0,5);

  // Priors on persistence
  rho_a_raw ~ normal(0, 0.5);
  rho_w_raw ~ normal(0, 0.5);
  rho_v_raw ~ normal(0, 0.5);

  // Priors on innovation scales
  log_sigma_a ~ normal(-1.203973,0.5);
  log_sigma_w ~ normal(-1.203973,0.5);
  log_sigma_v ~ normal(-1.203973,0.5);

  // Non-centered innovations
  a_z ~ std_normal();
  w_z ~ std_normal();
  v_z ~ std_normal();

  t0_raw ~ std_normal();

  for (n in 1:N) {
    if(yi[n]==1){
      real v_trial = stim[n]*v[n];
      if(choice[n] == 1){
        y[n] ~ wiener(a[n], t0, w[n], v_trial);
      }else{
        y[n] ~ wiener(a[n], t0, 1 - w[n], -v_trial);
      }
    }
  }
}
