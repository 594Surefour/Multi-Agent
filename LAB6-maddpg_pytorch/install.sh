# Install OpenAI Baselines 
git clone https://github.com.cnpmjs.org/openai/baselines.git
cd baselines
git checkout 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
pip install -e .
cd ..

# Install Environments
git clone https://github.com.cnpmjs.org/shariqiqbal2810/multiagent-particle-envs.git
cd multiagent-particle-envs/ 
pip install -e .
cd ..

# Install Other Dependancies
pip install torch==1.4.0 gym==0.9.4 tensorboard tensorboardX seaborn
