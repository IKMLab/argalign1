data {
  int<lower=0> num_illocutions;
  int<lower=0> num_categories;
  int<lower=0> num_observations;
  int<lower=0> num_commenters;
  int<lower=0> illocution[num_observations];
  int<lower=0> category[num_observations];
  int<lower=0> commenter[num_observations];

  int<lower=0> n_base[num_observations];
  int<lower=0> c_base[num_observations];

  int<lower=0> n_align[num_observations];
  int<lower=0> c_align[num_observations];

  real<lower=0> std_dev;
}

parameters {
  real eta_category_base[num_categories];
  real eta_illocution_align[num_illocutions];
  vector[num_categories] eta_illocution_base[num_illocutions];
  vector[num_illocutions] eta_category_align[num_categories];
  vector[num_observations] eta_observation_base;
  vector[num_observations] eta_observation_align;
}

transformed parameters {
  vector<lower=0,upper=1>[num_observations] mu_base;
  vector<lower=0,upper=1>[num_observations] mu_align;

  for (ix in 1:num_observations) {
    mu_base[ix] = inv_logit(eta_observation_base[ix]);
    mu_align[ix] = inv_logit(eta_observation_align[ix] + eta_observation_base[ix]);
  }
}

model {
  eta_category_base ~ cauchy(0, 2.5);
  eta_illocution_align ~ normal(0, std_dev);

  for (ix in 1:num_illocutions) {
    eta_illocution_base[ix] ~ normal(eta_category_base, std_dev);
  }

  for(ix in 1:num_categories) {
    eta_category_align[ix] ~ normal(eta_illocution_align, std_dev);
  }

  for(ix in 1:num_observations) {
    eta_observation_base[ix] ~ normal(eta_illocution_base[illocution[ix], category[ix]], std_dev);
    eta_observation_align[ix] ~ normal(eta_category_align[category[ix], illocution[ix]], std_dev);
  }

  c_base ~ binomial(n_base, mu_base);
  c_align ~ binomial(n_align, mu_align);
}