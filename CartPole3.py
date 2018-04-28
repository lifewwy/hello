import numpy as np
# import random
import gym
env = gym.make('CartPole-v0')

# Play with cross entropy agent, with given parameter vector
def crossEntropyAgent(num_episodes, max_episode_length, theta):
    rewards = []
    for i_episode in range(num_episodes):
        episode_reward = 0
        observation = env.reset()
        for t in range(max_episode_length):
            env.render()
            action = sampleAction(observation, theta)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                rewards.append(episode_reward)
                print("Reward for episode:", episode_reward)
                break

def trainCrossEntropyAgent(num_episodes, batch_size, max_episode_length, elite_fraction):
    # Initialise mu and sigma.
    mu = [0,0,0,0]
    sigma = [1,1,1,1]
    # env.monitor.start('/tmp/cartpole-experiment-2')
    for i_episode in range(num_episodes):
        print("Episode: ", i_episode)
        batch_rewards = []
        batch_theta = []
        for i_batch in range(batch_size):
            theta = sampleTheta(mu, sigma)
            observation = env.reset()
            batch_reward = 0
            for t in range(max_episode_length):
                env.render()
                action = sampleAction(observation, theta)
                observation, reward, done, info = env.step(action)
                batch_reward += reward
                if done:
                    batch_rewards.append(batch_reward)
                    batch_theta.append(theta)
                    break

        # Print the average reward
        print("Average rewards", np.mean(batch_rewards))

        # Now keep the top elite_fraction fraction of parameters theta, as
        # measured by reward
        indices = np.argsort(np.array(batch_rewards))
        indices = indices[::-1]

        elite_set = []
        cull_num = int(elite_fraction * len(indices))
        for i in range(cull_num):
            elite_set.append(batch_theta[indices[i]])

        # Now fit a diagonal Gaussian to this sample set, and repeat.
        [mu, sigma2] = fitGaussianToSamples(elite_set)
        sigma = np.sqrt(sigma2)

    # env.monitor.close()
    # Finally, return the mean we find
    return mu

def sampleTheta(mu, sigma):
    return np.random.randn(1,4) * sigma + mu

# observation is a row vector, and theta is a column vector.
# We want to find theta such that observation . theta > 0 is a good predictor
# for the 'move right' action.
def sampleAction(observation, theta):
    if np.dot(observation, np.transpose(theta)) > 0:
        return 1
    else:
        return 0

# Given a matrix whose rows are samples from a multivariate gaussian
# distribution with diagonal covariance matrix, we compute the maximum
# likelihood mean and covariance matrix. In fact we just return the diagonal of
# the covariance matrix.
def fitGaussianToSamples(samples):
    M = np.matrix(np.array(samples))
    Mshape = np.shape(M)
    numSamples = Mshape[0]
    numVariables = Mshape[1]

    # For each variable, we compute the mean and variance of the samples.
    mu = []
    sigma2 = []

    for i in range(numVariables):
        variableI = M[:,i]
        mu.append(np.mean(variableI))
        sigma2.append(np.var(variableI))
    return [mu, sigma2]

theta = trainCrossEntropyAgent(5, 100, int(1e5), 0.1)

# We have now computed our parameter theta, and we run the agent to see how it
# does with a time limit of 100,000 steps.
print("Theta after training", theta)

crossEntropyAgent(10, int(1e5), theta)