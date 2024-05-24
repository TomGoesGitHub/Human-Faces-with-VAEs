import tensorflow as tf
import tensorflow_probability as tfp

@tf.keras.utils.register_keras_serializable()
class AutoEncoder(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.built = True # layers are expected to be specified and intialized outside of this class

    def encode(self, input):
        latent_repr = self.encoder(input)
        return latent_repr # aka encoded
    
    def decode(self, latent_repr):
        decoded = self.decoder(latent_repr)
        return decoded

    def call(self, inputs, training):
        latent_repr = self.encode(inputs)
        output = self.decode(latent_repr)
        return output

    def get_latent_representation(self, input):
        return self.encode(input)

    def get_reconstruction(self, latent):
        return self.decode(latent)
    

@tf.keras.utils.register_keras_serializable()
class VariationalAutoEncoder(AutoEncoder):
    def __init__(self,
                 encoder,
                 decoder,
                 prior_distribution,
                 n_monte_carlo_samples = 5):
        
        super().__init__(encoder, decoder)
        self.prior_distribution = prior_distribution # prior p(z), non-trainable
        self.n_monte_carlo_samples = n_monte_carlo_samples # called L in Kingma-paper
        
        input_shape = encoder.layers[0].input_shape
        self.build(input_shape)
        self.compute_output_shape(input_shape)
        # note: it seems like this is required to be able to save the model
        # https://stackoverflow.com/questions/69311861/tf2-6-valueerror-model-cannot-be-saved-because-the-input-shapes-have-not-been
        
        self.summary()
        
    def encode(self, inputs):
        surrogate_posterior_distribution = self.encoder(inputs)
        return surrogate_posterior_distribution

    def decode(self, latent_repr):
        likelihood_distribution = self.decoder(latent_repr)
        return likelihood_distribution

    def get_latent_representation(self, input):
        dist = self.encode(input)
        latent = dist.sample()
        return latent
    
    def get_reconstruction(self, latent):
        dist = self.decode(latent)
        reconstruction = dist.sample()
        return reconstruction
    
    @tf.function
    def call(self, inputs):
        # encoding (note: in graph mode the sample will be computed directly)
        latent_repr = self.encode(inputs)

        # decoding (note: in graph mode the sample will be computed directly)
        reconstructed = self.decode(latent_repr)
        return reconstructed

    @tf.function
    def compute_elbo(self, x):
        PARALLEL_ITERS = 20
        surrogate_posterior_distribution = self.encode(x)
        latent_repr = surrogate_posterior_distribution.sample(self.n_monte_carlo_samples)
        log_likelihood_samples = tf.map_fn(lambda z: self.decode(z).log_prob(x),
                                           elems=latent_repr,
                                           parallel_iterations=PARALLEL_ITERS)
        log_likelihood_estimate = tf.reduce_mean(log_likelihood_samples, axis=0)

        try:
            # closed form KL
            # https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/kl_divergence
            kl = tfp.distributions.kl_divergence(surrogate_posterior_distribution, self.prior_distribution)
        except NotImplementedError:
            # KL-approximation with Monte-Carlo-sampling
            log_prior_samples = tf.map_fn(lambda z : self.prior_distribution.log_prob(z),
                                            elems=latent_repr,
                                            parallel_iterations=PARALLEL_ITERS)
            log_prior_estimate = tf.reduce_mean(log_prior_samples, axis=0)
            log_posterior_samples = tf.map_fn(lambda z : surrogate_posterior_distribution.log_prob(z),
                                                elems=latent_repr, 
                                                parallel_iterations=PARALLEL_ITERS)
            log_posterior_estimate = tf.reduce_mean(log_posterior_samples, axis=0)
            kl = log_posterior_estimate - log_prior_estimate
        
        elbo = tf.reduce_mean(-kl + log_likelihood_estimate)
        return elbo, kl, log_likelihood_estimate

    
    @tf.function
    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            elbo, kl, log_likelihood_estimate = self.compute_elbo(x)
            negative_elbo = (-1) * elbo
        
        gradients = tape.gradient(negative_elbo, self.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, self.trainable_variables))        
        return {**self.measure_performance_elbo(elbo, kl, log_likelihood_estimate),
                **self.measure_performance_reconstruction(x)
                }

    def measure_performance_elbo(self, elbo, kl, log_likelihood_estimate):
        return {'elbo': elbo,
                'loss': (-1)*elbo,
                'kl': tf.reduce_mean(kl),
                'log p(x|z)': tf.reduce_mean(log_likelihood_estimate)
               }
    
    def measure_performance_reconstruction(self, x):
        reconstructed = self.call(x)
        return {'reconstruction_loss': tf.reduce_mean((reconstructed - x)**2),
                'reconstruction_min': tf.reduce_min(reconstructed),
                'reconstruction_max': tf.reduce_max(reconstructed),
               }

    def test_step(self, data):
        x, _ = data
        elbo, kl, log_likelihood_estimate = self.compute_elbo(x)
        return {**self.measure_performance_elbo(elbo, kl, log_likelihood_estimate),
                **self.measure_performance_reconstruction(x)
                }

    def get_config(self):
        # https://www.tensorflow.org/guide/keras/serialization_and_saving
        base_config = super().get_config()
        config = {
            'encoder': tf.keras.utils.serialize_keras_object(self.encoder),
            'decoder': tf.keras.utils.serialize_keras_object(self.decoder),
            'prior_distribution': self.prior_distribution,
            'n_monte_carlo_samples': self.n_monte_carlo_samples,
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        # https://www.tensorflow.org/guide/keras/serialization_and_saving
        prior_distribution = config.pop('prior_distribution')
        n_monte_carlo_samples = config.pop('n_monte_carlo_samples')
        encoder = tf.keras.utils.deserialize_keras_object(config.pop('encoder'))
        decoder = tf.keras.utils.deserialize_keras_object(config.pop('decoder'))
        return cls(encoder, decoder, prior_distribution, n_monte_carlo_samples)


# class SparseAutoEncoder(AutoEncoder):
#     def __init__(self, encoder, decoder, l1=0, rho_KL=0, beta_l1=0, beta_KL=0, input_noise_layer=None):
#         super().__init__(encoder, decoder)
#         self.input_noise_layer = input_noise_layer
#         self.loss_tracker = tf.keras.metrics.Mean(name='loss')
#         self.l1 = l1
#         self.rho_KL = rho_KL

#         self.kl_div = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
#         self.beta_l1 = beta_l1
#         self.beta_KL = beta_KL
    
#         def __call__(self, inputs, training):
#             if training:
#                 if self.input_noise_layer:
#                     inputs = self.input_noise_layer(inputs)
#             latent_repr = self.encoder(inputs)
#             output = self.decoder(latent_repr)
#             return output
    
#     def compute_loss(self, x, y, y_pred, sample_weight):
#         encoded = self.encoder(x) # todo: this implementation requires another forward pass
#         reconstruction_loss = self.loss(y, y_pred)
        
#         # l1 regularization
#         regularization_loss_l1 = self.l1 * tf.reduce_sum(tf.math.abs(encoded), axis=-1)
        
#         # kl regulariztation
#         rho_hat = tf.reduce_mean(encoded, axis=0)
#         regularization_loss_KL = self.beta_KL * self.kl_div([1-self.rho_KL, self.rho_KL],
#                                                             [1-rho_hat, rho_hat])
        
#         # contraction loss
#         jac = 
#         contraction_loss = 

#         loss = reconstruction_loss + regularization_loss_l1 + regularization_loss_KL
#         self.loss_tracker.update_state(loss)
#         return loss

#     @property
#     def metrics(self):
#         return [self.loss_tracker]