import os
import itertools

os.system('rm -rf inner_run_bash/baseline')
os.mkdir('inner_run_bash/baseline')
os.chdir('inner_run_bash/baseline')


def name(x):
    return ('-'.join(x))


def genreate(
        rl_algo,
        cuda,
        game,
        idx,
        dir='../..',
):
    templete = f'''
    p=$(dirname $0)
    cd $p/{dir}/    



     CUDA_VISIBLE_DEVICES={cuda}  python examples.py --rl_algo {rl_algo} --cuda 0 --game {game}  --log_name {idx} &
     sleep 1s


        '''

    return templete


def generate_config(**kwargs):
    all_keys = list(kwargs.keys())
    all_values = list(kwargs.values())
    all_res_keys = [x[:-1] for x in all_keys]
    res = itertools.product(*all_values)
    res = [dict(zip(all_res_keys, x)) for nx, x in enumerate(res)]
    # for nx, x in enumerate(res):
    #     x['setting'] = nx
    return (res)


games = ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
         'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
         'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
         'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
         'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
         'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
         'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
         'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
         'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']

gym_games = [''.join([g.capitalize() for g in game.split('_')]) + 'NoFrameskip-v4' for game in games]

idxs = list(range(3))

cudas = [0]

rl_algos = ['quantile_regression_dqn_pixel', 'categorical_dqn_pixel', 'a2c_pixel', 'dqn_pixel', 'ppo_pixel']

config_pool = {
    'games': gym_games,
    'idxs': idxs,
    'cudas': cudas,
    'rl_algos' : rl_algos
}
all_res = generate_config(**config_pool)


class bash_generator():
    def __init__(self):
        self.gpu_count = 0
        self.all_job_num = 0
        self.job_log = open('job_log.csv', 'w')
        self.job_num = 0

        self.set_config()
        self.init_bash()
        self.init_fp()
        self.init_joblog()

    def add_one(self, job):
        self.all_job_num += 1
        job_config_string = name(job)
        print(self.gpu_count, job['cuda'])
        self.job_log.write(',{},{}\n'.format(self.all_job_num, job_config_string))
        job['cuda'] = self.gpu_count
        self.bash.append(genreate(**job))

        self.gpu_count += 1

        if self.gpu_count == self.card_per_job:
            self.write_fp()
            self.init_joblog()
            self.init_bash()
            self.init_fp()
            self.gpu_count = 0

    def set_config(self, card_per_job=4):

        self.card_per_job = card_per_job

    def init_bash(self):
        self.bash = []
        self.dir_tmp = []
        # self.bash.append('cd /philly/eu2/pnrsy/v-yuewng/project/philly-dqn-baseline/dqn_baseline')
        # self.bash.append('nvidia-smi --loop=2 &')

    def init_fp(self):
        self.fp = open('%d_job_%d.sh' % (self.job_num, self.all_job_num), 'w')
        self.job_num += 1

    def init_joblog(self):
        self.job_log.write('job_%d, \n' % (self.all_job_num))

    def write_fp(self):
        if self.bash:
            self.fp.write('\n'.join(self.bash))
            self.fp.write('\n wait')
        self.fp.close()

    def close_joblog(self):
        self.job_log.close()


a = bash_generator()
[a.add_one(x) for x in all_res]
a.write_fp()
a.close_joblog()
