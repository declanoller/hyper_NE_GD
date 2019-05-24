import RunTools as rt
from classes.EvoAgent import EvoAgent
from classes.HyperEPANN import HyperEPANN
import os
import shutil
import numpy as np
from math import sqrt, floor
import matplotlib.pyplot as plt
import FileSystemTools as fst
import traceback as tb

from movieMaker import imgsToVid




class Population:


    def __init__(self, **kwargs):

        self.agent_class = kwargs.get('agent_class', None)
        assert self.agent_class is not None, 'Need to provide an agent class! exiting'
        self.agent_class_name = self.agent_class.__name__
        self.gym_env = None

        self.init_kwargs = kwargs

        if self.agent_class_name == 'GymAgent':
            self.agent_class_str = kwargs.get('env_name', None)
            assert self.agent_class_str is not None, 'Need to provide an env if using GymAgent!'
        else:
            self.agent_class_str = self.agent_class_name

        # Pop properties
        self.N_pop = int(kwargs.get('N_pop', 15))
        self.mutate_std = kwargs.get('std', 1.0)
        self.best_N_frac = kwargs.get('best_N_frac', 1/5.0)

        # Run dir stuff
        self.fname_notes = kwargs.get('fname_notes', '')
        self.datetime_str = fst.getDateString()
        self.base_dir = kwargs.get('base_dir', 'output/misc_runs')

        self.dir = os.path.join(self.base_dir, 'evolve_{}_{}'.format(self.agent_class_str, self.datetime_str))
        fst.makeDir(self.dir)
        self.plot_dir = fst.makeDir(os.path.join(self.dir, 'plots'))
        self.parent_dir = fst.makeDir(os.path.join(self.dir, 'parents'))
        print('run dir: ', self.dir)
        pop_kwargs = {'run_dir' : self.dir}

        # Stuff specific to GymAgent
        if self.agent_class_name == 'GymAgent':
            self.createGymEnv()
            pop_kwargs['env_obj'] = self.gym_env

        both_kwargs = {**kwargs, **pop_kwargs}

        self.population = [EvoAgent(**both_kwargs) for i in range(self.N_pop)]



    def init_evo_data(self, grad_desc):

        # This sets whether we're using gradient descent or not.
        self.grad_desc = grad_desc

        self.best_FFs = []
        self.mean_FFs = []

        self.all_FFs = []
        self.all_nodecounts = []
        self.all_weightcounts = []
        self.champion_FF_mean_std = [] # A list of pairs of [mean, std] for runs of the current champion.

        best_N = max(int(self.N_pop*self.best_N_frac), 2)
        # Should all be the same at this point.
        self.parent_indices = [list(range(best_N))]
        fname = os.path.join(self.parent_dir, 'NN__gen_{}__parent_{}.json')
        [self.population[i].saveNetworkToFile(fname=fname.format(0, i)) for i in range(best_N)]

        self.reset_generation_evo_data()


    def reset_generation_evo_data(self):
        ############# Things that get reset each generation:

        self.generation_best_FF = -100000000
        self.generation_mean_FF = 0


        self.mean_Rs = []
        if self.grad_desc:
            self.train_curves = []


    def add_agent_episodes_data(self, agent_index, episodes_ret):

        self.mean_Rs.append([agent_index, episodes_ret['agent_mean_episode_score']])
        if self.grad_desc:
            self.train_curves.append(episodes_ret['train_curve'])

        self.generation_mean_FF += episodes_ret['agent_mean_episode_score']
        if episodes_ret['agent_mean_episode_score'] > self.generation_best_FF:
            self.generation_best_FF = episodes_ret['agent_mean_episode_score']


    def update_evo_data(self):
        self.generation_mean_FF = np.mean(self.generation_mean_FF)
        self.best_FFs.append(self.generation_best_FF)
        self.mean_FFs.append(self.generation_mean_FF)

        self.mean_Rs_no_label = [x[1] for x in self.mean_Rs]

        self.all_FFs.append(self.mean_Rs_no_label)
        self.all_nodecounts.append([x.getNAtoms() for x in self.population])
        self.all_weightcounts.append([x.getNConnections() for x in self.population])


    def print_save_evo_data(self, i):

        print('\ngen {:.1f}. Best FF = {:.4f}, mean FF = {:.4f}'.format(i, self.generation_best_FF, self.generation_mean_FF))
        print('avg network size: {:.3f}'.format(np.mean([x.getNAtoms() for x in self.population])))
        print('avg # network connections: {:.3f}'.format(np.mean([x.getNConnections() for x in self.population])))

        self.plotPopHist(self.mean_Rs_no_label, 'pop_FF')

        fname = os.path.join(self.plot_dir, '{}_gen_{}.png'.format('bestNN', i))
        best_agent_index = self.sortByFitnessFunction(self.mean_Rs)[0][0]
        self.population[best_agent_index].plotNetwork(  show_plot=False,
                                                        save_plot=True,
                                                        fname=fname,
                                                        atom_legend=False)

        self.population[best_agent_index].eval_agent_save(
                                            fname=os.path.join(self.plot_dir, '{}_gen_{}.txt'.format('bestNN_output', i)))
        if self.grad_desc:
            self.plot_train_curve(self.train_curves[best_agent_index], f'train_curve_gen_{i}.png')


    def update_evo_champion_data(self, N_runs_each_champion):

        ########
        # Not using this for now, I dont really care about it.

        champion_scores = []
        for run in range(N_runs_each_champion):
            champion_scores.append((self.population[0].runEpisode(N_eval_steps)['r_avg']))

        champion_FF_mean_std.append([np.mean(champion_scores), np.std(champion_scores)])


    def save_final_evo_data(self, **kwargs):

        # Pretty messy but I just wanted to get it the hell
        # out of the main evolve() function.

        grad_desc = kwargs.get('grad_desc', True)
        N_trials_per_agent = kwargs.get('N_trials_per_agent', 3)
        N_eval_steps = kwargs.get('N_eval_steps', 400)
        N_gen = kwargs.get('N_gen', 50)
        stop_gen = N_gen
        N_trials_per_agent = kwargs.get('N_trials_per_agent', 3)
        N_runs_each_champion = kwargs.get('N_runs_each_champion', 5)
        N_runs_with_best = kwargs.get('N_runs_with_best', 10)
        assert N_runs_with_best > 0, 'Need at least one run with best individ!'
        record_final_runs = kwargs.get('record_final_runs', False)
        show_final_runs = kwargs.get('show_final_runs', False)
        reward_stop_goal = kwargs.get('reward_stop_goal', None)


        self.saveScore(self.best_FFs, 'bestscore')
        self.saveScore(self.mean_FFs, 'meanscore')
        self.saveScore(self.all_FFs, 'all_FFs')
        self.saveScore(self.all_nodecounts, 'nodecounts')
        self.saveScore(self.all_weightcounts, 'weightcounts')
        self.saveScore(self.champion_FF_mean_std, 'champion_FF_mean_std')

        # Plot best and mean FF curves for the population
        plt.subplots(1, 1, figsize=(8,8))
        plt.plot(self.mean_FFs, color='dodgerblue', label='Pop. avg FF')
        plt.plot(self.best_FFs, color='tomato', label='Pop. best FF')
        plt.xlabel('generations')
        plt.ylabel('FF')
        plt.legend()
        fname = os.path.join(self.dir, '{}_{}.png'.format('FFplot', self.datetime_str))
        plt.savefig(fname)

        plt.close()



        # Get an avg final score for the best individ. You know this will be the best one because
        # the best one is preserved after getNextGen().
        best_individ = self.population[0]

        # Save the NN of the best individ.
        bestNN_fname = os.path.join(self.dir, f'bestNN_{self.agent_class.__name__}_{self.datetime_str}')
        best_individ.saveNetworkToFile(fname=(bestNN_fname + '.json'))
        best_individ.plotNetwork(show_plot=False, save_plot=True, fname=(bestNN_fname + '.png'), node_legend=True)

        self.saveIndividualAsAtom(best_individ)

        # Something annoying happening with showing vs recording the final runs, but I'll figure it out later.
        best_individ_scores = [best_individ.runEpisode(
            show_episode=show_final_runs,
            record_episode=record_final_runs,
            **kwargs)['r_avg'] for i in range(N_runs_with_best)]
        #best_individ.agent.closeEnv()
        best_individ_avg_score = np.mean(best_individ_scores)

        fname = os.path.join(self.parent_dir, 'parents.txt')
        np.savetxt(fname, self.parent_indices, delimiter='\t', fmt='%d')


        # Plot some more stuff with the saved dat
        try:
            rt.plotPopulationProperty(self.dir, 'all_FFs', make_hist_gif=False)
            rt.plotPopulationProperty(self.dir, 'weightcounts', make_hist_gif=False)
            rt.plotPopulationProperty(self.dir, 'nodecounts', make_hist_gif=False)
        except:
            print('\n\n')
            print(tb.format_exc())
            print('plotPopulationProperty() failed, continuing')


        try:
            if record_final_runs:
                import movie_combine
                N_side = min(3, floor(sqrt(N_runs_with_best)))
                movie_dir = best_individ.agent.record_dir
                movie_combine.combineMovieFiles(path=movie_dir, grid_size=f'{N_side}x{N_side}', make_gif=True)
        except:
            print('failed combining movies into single panel')

        return_dict = {}
        return_dict['best_FFs'] = self.best_FFs
        return_dict['mean_FFs'] = self.mean_FFs
        return_dict['best_individ_avg_score'] = best_individ_avg_score

        return(return_dict)
        '''# Plot the mean and std for the champion of each generation
        champion_FF_mean_std = np.array(self.champion_FF_mean_std)
        champ_mean = champion_FF_mean_std[:,0]
        champ_std = champion_FF_mean_std[:,1]
        plt.fill_between(np.array(range(len(champ_mean))), champ_mean - champ_std, champ_mean + champ_std, facecolor='dodgerblue', alpha=0.5)
        plt.plot(champ_mean, color='mediumblue')
        plt.xlabel('generations')
        plt.ylabel('FF')
        fname = os.path.join(self.dir, '{}_{}.png'.format('champion_mean-std_plot', self.datetime_str))
        plt.savefig(fname)'''


    def evolve(self, **kwargs):

        ##### kwargs stuff

        grad_desc = kwargs.get('grad_desc', True)
        N_trials_per_agent = int(kwargs.get('N_trials_per_agent', 3))
        N_eval_steps = int(kwargs.get('N_eval_steps', 20))
        N_gen = int(kwargs.get('N_gen', 50))
        stop_gen = N_gen
        max_runtime = kwargs.get('max_runtime', 60)

        N_runs_each_champion = int(kwargs.get('N_runs_each_champion', 5))
        N_runs_with_best = int(kwargs.get('N_runs_with_best', 10))
        assert N_runs_with_best > 0, 'Need at least one run with best individ!'
        record_final_runs = kwargs.get('record_final_runs', False)
        show_final_runs = kwargs.get('show_final_runs', False)
        reward_stop_goal = kwargs.get('reward_stop_goal', None)


        ########### Run specific stuff

        start_time = fst.getCurTimeObj()

        # Create a log file for the kwargs
        log_fname = os.path.join(self.dir, f'log_{self.datetime_str}.txt')
        fst.writeDictToFile({**self.init_kwargs, **kwargs}, log_fname)


        ############ Lists for collecting info

        self.init_evo_data(grad_desc)

        overtime = False

        for i in range(N_gen):


            runtime = fst.getTimeDiffNum(start_time)
            if runtime > max_runtime:
                print('\n\nReached max runtime of {:.2f} s at gen {}/{}, stopping here.'.format(runtime, i+1, N_gen))
                overtime = True
                break

            self.reset_generation_evo_data()

            for agent_index, individ in enumerate(self.population):
                episodes_ret = individ.run_N_episodes(**kwargs)
                self.add_agent_episodes_data(agent_index, episodes_ret)

            self.update_evo_data()

            if (i%max(1, int(N_gen/20))==0):
                self.print_save_evo_data(i)

            self.getNextGen(self.mean_Rs)

            #self.update_evo_champion_data()

            if reward_stop_goal is not None:
                if i>=stop_gen:
                    print(f'reached generation {stop_gen} : stopping.')
                    break

                if (self.generation_best_FF >= reward_stop_goal) and (stop_gen >= N_gen):
                    # Will make it go 5% further than it has so far, just to make a nicer
                    # looking plot.
                    print(f'reward_stop_goal ({reward_stop_goal}) reached: {self.generation_best_FF}, in gen {i}')
                    stop_gen = int(1.15*i)
                    print('continuing slightly to generation', stop_gen)
                    self.print_save_evo_data(i)


        final_sorted = self.sortByFitnessFunction(self.mean_Rs)

        runtime = fst.getTimeDiffNum(start_time)
        print('\n\nRun took: {}\n\n'.format(fst.getTimeDiffStr(start_time)))

        return_dict = self.save_final_evo_data(**kwargs)
        return_dict['runtime'] = runtime
        return_dict['overtime'] = overtime
        return(return_dict)


    def getNextGen(self, FF_list):

        pop_indices_sorted = self.sortByFitnessFunction(FF_list)
        best_N = max(int(self.N_pop*self.best_N_frac), 2)
        #best_N = 1
        top_N_indices = [x[0] for x in pop_indices_sorted[:best_N]]
        #print('best individ:')
        #self.population[top_N_indices[0]].printNetwork()

        gen = len(self.parent_indices)
        fname = os.path.join(self.parent_dir, 'NN__gen_{}__parent_{}.json')
        [self.population[top_N_indices[i]].saveNetworkToFile(fname=fname.format(gen, i)) for i in range(best_N)]

        self.parent_indices.append([top_N_indices[i]%best_N for i in range(best_N)])

        #new_pop = [self.population[top_N_indices[0]].clone()]
        new_pop = []
        mod_counter = 0

        while len(new_pop)<self.N_pop:

            new_EPANN = self.population[top_N_indices[mod_counter%best_N]].clone()
            new_EPANN.mutate(std=self.mutate_std)

            new_pop.append(new_EPANN)
            mod_counter += 1

        self.population = new_pop


    def saveScore(self, score, label):
        fname = os.path.join(self.dir, '{}_{}.txt'.format(label, self.datetime_str))
        np.savetxt(fname, score, fmt='%.4f')


    def sortByFitnessFunction(self, FF_list):

        # Assumes self.pop is in the same order it was when it was evaluated.

        pop_indices_sorted = sorted(FF_list, key=lambda x: -x[1])
        return(pop_indices_sorted)


    def saveIndividualAsAtom(self, individ):

        #
        atom_save_fname = os.path.join(self.dir, f'Atom_{self.agent_class_name}_{self.datetime_str}.json')
        atom_name = self.agent_class_str
        individ.saveNetworkAsAtom(fname=atom_save_fname, atom_name=atom_name)


    def plotPopHist(self, hist_var, label):

        # Stuff for plotting topo change stuff
        #self.plotPopHist([len(epann.node_list) for epann in self.population], 'pop_nodecount')
        #self.plotPopHist([len(epann.weights_list) for epann in self.population], 'pop_weightcount')

        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        ax.hist(hist_var, facecolor='dodgerblue', edgecolor='k', label=label)
        plt.axvline(np.mean(hist_var), color='k', linestyle='dashed', linewidth=1)
        plt.xlabel(label)
        ax.legend()
        fname = os.path.join(self.plot_dir, '{}_{}.png'.format(label, fst.getDateString()))
        plt.savefig(fname)

        # Is this the way to make sure too many figures don't get created? or plt.close()?
        plt.close()


    def plot_train_curve(self, in_dat, rel_fname):
        plt.clf()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.plot(in_dat)
        plt.savefig(os.path.join(self.plot_dir, rel_fname))


    def createGymEnv(self):

        # Do this so it only creates one gym env for the whole pop, which should
        # make everything neater and maybe faster.

        self.gym_env_individ = self.agent_class(**self.init_kwargs)
        self.gym_env = self.gym_env_individ.env


    def __del__(self):
        if self.gym_env is not None:
            self.gym_env.close()



def multi_run_hist(**kwargs):

    fname = kwargs.get('fname', 'hist_test')
    N_evo_runs = kwargs.get('N_evo_runs', 10)

    datetime_str = fst.getDateString()
    base_dir = os.path.join('output/multi_run_output/', f'multi_{datetime_str}')
    os.mkdir(base_dir)
    params_fname = os.path.join(base_dir, f'params_{datetime_str}.txt')
    kwargs = {**kwargs, 'base_dir' : base_dir}
    fst.writeDictToFile(kwargs, params_fname)

    print('\n\nStarting new multi run...\n\n')
    start_time = fst.getCurTimeObj()

    overtime_count = 0
    runtime_dist = []
    for i in range(N_evo_runs):
        print(f'\n\niteration {i}\n\n')
        p1 = Population(**kwargs)
        d = p1.evolve(**kwargs)
        runtime_dist.append(d['runtime'])
        if d['overtime']:
            overtime_count += 1


    print(f'\n\n\n{overtime_count}/{N_evo_runs} runs went overtime.\n')
    runtime = fst.getTimeDiffNum(start_time)
    print('\n\nMulti run took: {}\n\n'.format(fst.getTimeDiffStr(start_time)))


    mean_runtime = np.mean(runtime_dist)
    #plt.clf()
    plt.close('all')
    plt.hist(runtime_dist)

    ax = plt.gca()
    ax.axvline(mean_runtime, color='red', linestyle='dashed')

    plt.xlabel('Run times (s)')
    hist_fname = os.path.join(base_dir, f'hist_{datetime_str}.png')
    plt.savefig(hist_fname)
    #plt.show()
    if kwargs.get('negate_runtime', False):
        return(-np.mean(runtime_dist))
    else:
        return(np.mean(runtime_dist))





def create_lineage_movie(dir, **kwargs):

    par_ind_fname = os.path.join(dir, 'parents.txt')
    parent_indices = np.loadtxt(par_ind_fname, dtype='int')

    lineage_dir = os.path.join(dir, 'pics')

    if os.path.exists(lineage_dir):
        shutil.rmtree(lineage_dir)

    os.mkdir(lineage_dir)


    nn = HyperEPANN()
    fname = os.path.join(dir, 'NN__gen_{}__parent_{}.json')
    plot_ind = 0
    for gen in range(parent_indices.shape[0]-1, -1, -1):
        nn.loadNetworkFromFile(fname=fname.format(gen, plot_ind))
        nn.plotNetwork(
                        show_plot=False,
                        save_plot=True,
                        fname=os.path.join(lineage_dir, f'{gen}.png'),
                        plot_title=f'Generation {gen}')
        plot_ind = parent_indices[gen][plot_ind]

    fname = kwargs.get('fname', '')
    movie_fname = os.path.join(dir, 'NN_{}_evo_mov.mp4'.format(fname))
    if os.path.exists(movie_fname):
        os.remove(movie_fname)

    mov_length_s = kwargs.get('mov_length', 5)
    imgsToVid(lineage_dir, movie_fname, framerate=parent_indices.shape[0]/mov_length_s, crf=22)






#
