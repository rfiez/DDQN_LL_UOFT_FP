This is the executable python file: ddqn_finalproj_3547_Rueen_Fiez.py

Solved for an episode is defined as reaching a reward of +200. We track the Agent’s average rewards across a span of 100 episodes. Our solution uses 500 episodes for testing purposes where an average reward of 50-75 should serve as an acceptable solved baseline. This is our main objective for the DDQN project.

Rewards scheme details are as follows:
-	Lander’s Leg ground contact:		+10 reward
-	Firing Rockets/Thrust Downward:	-0.3 reward
-	Crash landing is an additional:		-100 reward
-	Graceful landing is an additional:		+100 reward
-	Moving away from the goal position will reduce reward 
-	Moving closer to the goal position will increase reward 


1. Make sure to activate the virtual env "final_proj_v_env"
2. Make sure you have dependencies like swigwin3.0.12, ffmpeg, and pybox2d
3. Make sure to look at lines 36-60 for Global Variables and Hyperparameters
	and toggle their values as required for DEBUGGING, Hyperparams, MONITOR_AGENT, etc...
4. If you want video of all episodes compiled into one video see this param on line #44 DO_VIDEO_COMPILATION_OF_ALL_EPISODES_AT_END
5. Make sure you have the following as part of your env (see pip list below):
absl-py              0.9.0
astor                0.8.1
box2d-py             2.3.8
cachetools           4.0.0
certifi              2019.11.28
chardet              3.0.4
cloudpickle          1.3.0
cycler               0.10.0
decorator            4.4.2
ffmpeg               1.4
future               0.18.2
gast                 0.2.2
google-auth          1.11.3
google-auth-oauthlib 0.4.1
google-pasta         0.2.0
grpcio               1.27.2
gym                  0.17.1
h5py                 2.10.0
idna                 2.9
imageio              2.8.0
imageio-ffmpeg       0.4.1
Keras                2.3.1
Keras-Applications   1.0.8
Keras-Preprocessing  1.1.0
kiwisolver           1.1.0
Markdown             3.2.1
matplotlib           3.0.3
moviepy              1.0.1
natsort              7.0.1
numpy                1.18.2
oauthlib             3.1.0
opt-einsum           3.2.0
Pillow               7.0.0
pip                  20.0.2
proglog              0.1.9
protobuf             3.11.3
pyasn1               0.4.8
pyasn1-modules       0.2.8
pyglet               1.5.0
pyparsing            2.4.6
python-dateutil      2.8.1
PyYAML               5.3.1
requests             2.23.0
requests-oauthlib    1.3.0
rsa                  4.0
scipy                1.4.1
setuptools           46.1.1
six                  1.14.0
tensorboard          1.15.0
tensorflow           1.15.0
tensorflow-estimator 1.15.1
termcolor            1.1.0
tqdm                 4.43.0
urllib3              1.25.8
Werkzeug             1.0.0
wheel                0.34.2
wrapt                1.12.1
