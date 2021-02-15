import numpy as np
import collections
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import sts

from matplotlib import pylab as plt
from model.base_model import *


class modelSTS(baseModel):
    def __init__(self, steps: int, training_data: np.array):
        self.model = None
        # Allow external control of optimization to reduce test runtimes.
        self.num_variational_steps = steps  # @param { isTemplate: true}
        self.training_data = training_data.astype(np.float32)
        self.variational_posteriors = None
        self.optimizer = None
        self.surrogate_posterior = None
        self.samples = None
        self.forecast_dist = None
        pass

    def build_model_sts(self):
        trend = sts.LocalLinearTrend(
            observed_time_series=self.training_data,
            name="local_linear_trend"
        )
        semi_local = tfp.sts.SemiLocalLinearTrend(
            observed_time_series=self.training_data,
            name="semi_local_trend"
        )
        local_level = tfp.sts.LocalLevel(
            observed_time_series=self.training_data,
            name="local_level_trend"
        )
        twodays_seasonal = tfp.sts.Seasonal(
            num_seasons=2,
            observed_time_series=self.training_data,
            name="two_days_model"
        )
        threedays_seasonal = tfp.sts.Seasonal(
            num_seasons=3,
            observed_time_series=self.training_data,
            name="three_days_model"
        )
        fivedays_seasonal = tfp.sts.Seasonal(
            num_seasons=5,
            observed_time_series=self.training_data,
            name="five_days_model"
        )
        weekly_seasonal = tfp.sts.Seasonal(
            num_seasons=7,
            observed_time_series=self.training_data,
            name="weekly_model"
        )
        biweekly_seasonal = tfp.sts.Seasonal(
            num_seasons=14,
            observed_time_series=self.training_data,
            name="biweekly_model"
        )
        monthly_seasonal = tfp.sts.Seasonal(
            num_seasons=30,
            observed_time_series=self.training_data,
            name="monthly_model"
        )
        residual_level = tfp.sts.Autoregressive(
            order=1,
            observed_time_series=self.training_data,
            name='residual'
        )
        self.model = sts.Sum([trend,
                              semi_local,
                              local_level,
                              twodays_seasonal,
                              threedays_seasonal,
                              fivedays_seasonal,
                              weekly_seasonal,
                              biweekly_seasonal,
                              monthly_seasonal,
                              residual_level
                              ], observed_time_series=self.training_data)
        self.variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=self.model)
        self.optimizer = tf.optimizers.Adam(learning_rate=.1)

    def training_surrogate(self):
        self.surrogate_posterior = tfp.sts.build_factored_surrogate_posterior(model=self.model)

        elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=self.model.joint_log_prob(observed_time_series=self.training_data),
            surrogate_posterior=self.variational_posteriors,
            optimizer=self.optimizer,
            num_steps=self.num_variational_steps)
        return elbo_loss_curve

    def forcast_surrogate(self, sample=50):
        samples = self.surrogate_posterior.sample(sample)
        forecast_dist = tfp.sts.forecast(self.model, self.training_data,
                                         parameter_samples=samples,
                                         num_steps_forecast=50)

        forecast_mean = forecast_dist.mean()[..., 0]  # shape: [50]
        forecast_scale = forecast_dist.stddev()[..., 0]  # shape: [50]
        forecast_samples = forecast_dist.sample()[..., 0]  # shape: [10, 50]

        self.plot_forecast(self.training_data,
                           forecast_mean=forecast_mean,
                           forecast_scale=forecast_scale,
                           forecast_samples=forecast_samples)

    def training_hmc(self):
        self.samples, kernel_results = tfp.sts.fit_with_hmc(self.model,
                                                            observed_time_series=self.training_data,
                                                            num_results=30,
                                                            num_variational_steps=300)

        print("acceptance rate: {}".format(
            np.mean(kernel_results.inner_results.inner_results.is_accepted, axis=0)))
        print("posterior means: {}".format(
            {param.name: np.mean(param_draws, axis=0)
             for (param, param_draws) in zip(self.model.parameters, self.samples)}))

    def forcast_hmc(self, num_steps_forecast=10):
        self.forecast_dist = tfp.sts.forecast(self.model, self.training_data,
                                         parameter_samples=self.samples,
                                         num_steps_forecast=num_steps_forecast)

        forecast_mean = self.forecast_dist.mean()[..., 0]  # shape: [50]
        forecast_scale = self.forecast_dist.stddev()[..., 0]  # shape: [50]
        forecast_samples = self.forecast_dist.sample()[..., 0]  # shape: [10, 50]

        return forecast_mean, forecast_scale, forecast_samples

    def get_trend_dist(self):
        component_dists = sts.decompose_by_component(
            self.model,
            observed_time_series=self.training_data,
            parameter_samples=self.samples
        )
        forecast_component_dists = sts.decompose_forecast_by_component(
            self.model,
            forecast_dist=self.forecast_dist,
            parameter_samples=self.samples
        )
        component_means, component_stddevs = (
            {k.name: c.mean() for k, c in component_dists.items()},
            {k.name: c.stddev() for k, c in component_dists.items()})
        forecast_component_means, forecast_component_stddevs = (
            {k.name: c.mean() for k, c in forecast_component_dists.items()},
            {k.name: c.stddev() for k, c in forecast_component_dists.items()})
        for key in component_means:
            component_means[key] = component_means[key].numpy().tolist()
        for key in component_stddevs:
            component_stddevs[key] = component_stddevs[key].numpy().tolist()
        for key in forecast_component_means:
            forecast_component_means[key] = forecast_component_means[key].numpy().tolist()
        for key in forecast_component_stddevs:
            forecast_component_stddevs[key] = forecast_component_stddevs[key].numpy().tolist()

        return component_means, component_stddevs, forecast_component_means, forecast_component_stddevs

    def plot_forecast(self,
                      observed_time_series,
                      forecast_mean,
                      forecast_scale,
                      forecast_samples):
        plt.figure(figsize=(12, 6))

        num_steps = observed_time_series.shape[-1]
        num_steps_forecast = forecast_mean.shape[-1]
        num_steps_train = num_steps - num_steps_forecast

        c1, c2 = (0.12, 0.47, 0.71), (1.0, 0.5, 0.05)
        plt.plot(np.arange(num_steps), observed_time_series,
                 lw=2, color=c1, label='ground truth')

        forecast_steps = np.arange(num_steps_train,
                                   num_steps_train + num_steps_forecast)
#        plt.plot(forecast_steps, forecast_samples, lw=1, color=c2, alpha=0.1)
        plt.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
                 label='forecast')
#        plt.fill_between(forecast_steps,
#                         forecast_mean - 2 * forecast_scale,
#                         forecast_mean + 2 * forecast_scale, color=c2, alpha=0.2)

        plt.xlim([0, num_steps])
        plt.legend()
        plt.show()
