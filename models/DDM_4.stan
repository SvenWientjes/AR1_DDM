data {
  int<lower=1> P;                         // Number of participants
  int<lower=1> N;                         // Number of trials (per participant)
  array[P,N] int<lower=0,upper=1> choice; // Upper (1) or lower (0) boundary
  array[P,N] real<lower=0>        rt;     // RTs
  vector<lower=0>[P]              min_rt; // Minimum observed RT (to constrain t0)
}

parameters {
  real a_raw;  // Boundary separation
  real t0_raw; // Non-decision time
  real w_raw;  // Starting bias
  real v_raw;  // Drift rate
  
  row_vector<lower=0>[4] sigma_s; 
  matrix[P,4]            z;
  //  1 real boundary;
  //  2 real t0;
  //  3 real bias;
  //  4 real drift;
}
model {
  // Priors (vaguely informative)
  a_raw  ~ normal(0,0.5); // Centered around 1
  t0_raw ~ std_normal(); 
  w_raw  ~ std_normal();
  v_raw  ~ student_t(3, 0, 1.5); 
  
  sigma_s ~ lognormal(-1.2, 0.3);
  to_vector(z) ~ std_normal();

  // Likelihood
  for(p in 1:P){
    
    real a_lin  = a_raw  + sigma_s[1] * z[p,1];
    real t0_lin = t0_raw + sigma_s[2] * z[p,2];
    real w_lin  = w_raw  + sigma_s[3] * z[p,3];
    real v_lin  = v_raw  + sigma_s[4] * z[p,4];

    real a  = exp(a_lin);
    real t0 = inv_logit(t0_lin) * min_rt[p];
    real w  = Phi(w_lin);
    real v  = v_lin;
    
    real w_flip = 1-w;
    real v_flip = -v;
    
    for(n in 1:N){
      real drift  = (choice[p,n] == 1) ? v : v_flip;
      real bias   = (choice[p,n] == 1) ? w : w_flip;
      rt[p,n] ~ wiener(a, t0, bias, drift);
    }
  }
}
