import pong
import submitted
import matplotlib.pyplot as plt
import numpy as np

#state_quantization = [10,10,2,2,10]
state_quantization = None
q_learner = submitted.deep_q(alpha=0.5, epsilon=0.08, gamma=0.99, nfirst=5)

#q_learner.load('test_model.pkl', train=True)

pong_game = pong.PongGame(learner=q_learner, visible=False, state_quantization=state_quantization)

games = 500

scores, q_achieved, q_states = pong_game.run(m_games=games, states=[])

q_learner.save('test_model.pkl')

fig = plt.figure(figsize=(14,9),layout='tight')
ax = [ fig.add_subplot(3,1,x) for x in range(1,4) ]
ax[0].plot(np.arange(0,len(scores)),np.log10(1+np.array(scores)))
ax[0].plot([0,games],np.log10([7,7]),'k--')
ax[0].set_title('Game scores')
ax[1].plot(np.arange(games-9),np.log10(1+np.convolve(np.ones(10)/10,scores,mode='valid')))
ax[1].plot([0,games],np.log10([7,7]),'k--')
ax[1].set_title('Game scores, average 10 consecutive games')
ax[2].plot(np.arange(0,len(q_achieved)),q_achieved)
ax[2].set_title('Q values of state achieved at each time')
ax[2].set_ylabel('Game number')
plt.savefig('test.jpg')
