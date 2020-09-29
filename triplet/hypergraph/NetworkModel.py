import torch.nn as nn
import torch.optim
from triplet.hypergraph.Utils import *
import time
from termcolor import colored
from multiprocessing import Process
from triplet.hypergraph.BatchTensorNetwork import BatchTensorNetwork

class NetworkModel(nn.Module):
    Iter = 0

    def __init__(self, fm, compiler, evaluator, model_path = 'best_model.pt'):
        super().__init__()
        self.fm = fm
        self.compiler = compiler
        self.all_instances = None
        self.all_instances_test = None
        self.networks = None
        self.networks_test = None
        self.evaluator = evaluator
        self.model_path = model_path
        self.check_every = None
        self.weight_decay = 0

    def set_model_path(self, path):
        self.model_path = path

    def get_instances(self):
        return self.all_instances

    def get_feature_manager(self):
        return self.fm

    def get_network_compiler(self):
        return self.compiler

    def split_instances_for_train(self, insts_before_split):
        print("#instances=", len(insts_before_split))
        insts = [None for i in range(len(insts_before_split) * 2)]

        k = 0
        for i in range(0, len(insts), 2):
            insts[i] = insts_before_split[k]
            insts[i + 1] = insts_before_split[k].duplicate()
            insts[i + 1].set_instance_id(-insts[i].get_instance_id())
            insts[i + 1].set_weight(-insts[i].get_weight())
            insts[i + 1].set_unlabeled()
            k = k + 1
        return insts


    def lock_it(self):
        gnp = self.fm.get_param_g()

        if gnp.is_locked():
            return

        gnp.finalize_transition()
        gnp.locked = True


    def learn_batch(self, train_insts, max_iterations, dev_insts, test_insts, optimizer = 'adam', batch_size = 10):

        insts_before_split = train_insts

        insts = self.split_instances_for_train(insts_before_split)
        self.all_instances = insts

        batches = self.fm.generate_batches(insts_before_split, batch_size)


        # label_networks = []
        # unlabel_networks = []
        # for i in range(0, len(self.all_instances), 2):
        #     label_networks.append(self.get_network(i))
        #     unlabel_networks.append(self.get_network(i + 1))
        if self.networks == None:
            self.networks = [None for i in range(len(insts))]

        batch_tensor_networks = []

        for batch_idx, batch in enumerate(batches):
            batch_input_seqs, batch_network_id_range = batch
            inst_ids = list(range(batch_network_id_range[0], batch_network_id_range[1]))


            batch_label_networks = [self.get_network(i * 2) for i in inst_ids]
            batch_unlabel_networks = [self.get_network(i * 2 + 1) for i in inst_ids]

            batch_tensor_label_networks = BatchTensorNetwork(self.fm, batch_idx * 2 + 0, batch_label_networks, batch_network_id_range)
            batch_tensor_unlabel_networks = BatchTensorNetwork(self.fm, batch_idx * 2 + 1, batch_unlabel_networks, batch_network_id_range)

            batch_tensor_networks.append((batch_tensor_label_networks, batch_tensor_unlabel_networks))


        self.touch_batch(self.all_instances, batch_tensor_networks, batch_size)


        self.lock_it()
        parameters = filter(lambda p: p.requires_grad, self.parameters())

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters)
        elif optimizer == 'sgd':
            print(colored("Using SGD: lr is: {}, L2 regularization is: {}".format(NetworkConfig.NEURAL_LEARNING_RATE, NetworkConfig.l2), 'yellow'))
            optimizer = torch.optim.SGD(parameters,lr=NetworkConfig.NEURAL_LEARNING_RATE, weight_decay=NetworkConfig.l2)
        else:
            print(colored('Unsupported optimizer:', 'red'), optimizer)
            return

        self.best_score = None

        if self.check_every == None:
            self.check_every = len(batches)


        print('Start Training...', flush=True)
        for iteration in range(max_iterations): #Epoch

            if iteration == 99:
                print()

            self.train()
            all_loss = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(batches):
                optimizer.zero_grad()
                self.zero_grad()

                batch_loss = 0

                batch_input_seqs, batch_network_id_range = batch
                nn_batch_output = self.fm.build_nn_graph_batch(batch_input_seqs)
                this_batch_size = nn_batch_output.shape[0]
                batch_tensor_label_networks, batch_tensor_unlabel_networks = batch_tensor_networks[batch_idx]
                batch_tensor_label_networks.nn_batch_output = nn_batch_output
                batch_tensor_unlabel_networks.nn_batch_output = nn_batch_output

                label_score = self.forward(batch_tensor_label_networks)
                unlabel_score = self.forward(batch_tensor_unlabel_networks)
                batch_loss = torch.sum(unlabel_score - label_score, dim = 0)
                all_loss += batch_loss.item()

                batch_loss.backward()
                optimizer.step()

                def eval():
                    start_time = time.time()
                    results = self.decode_batch(dev_insts, batch_size)
                    score = self.evaluator.eval(results)
                    end_time = time.time()
                    print("Dev  -- ", str(score), '\tTime={:.2f}s'.format(end_time - start_time), flush=True)

                    if self.best_score == None or score.larger_than(self.best_score):

                        if self.best_score == None:
                            self.best_score = score
                        else:
                            self.best_score.update_score(score)
                        self.save()

                        start_time = time.time()
                        self.decode(test_insts)
                        test_score = self.evaluator.eval(test_insts)
                        end_time = time.time()
                        print(colored("Test -- ", 'red'), str(test_score),
                              '\tTime={:.2f}s'.format(end_time - start_time), flush=True)

                    else:
                        if NetworkConfig.ECHO_TEST_RESULT_DURING_EVAL_ON_DEV:
                            start_time = time.time()
                            self.decode(test_insts)
                            test_score = self.evaluator.eval(test_insts)
                            end_time = time.time()
                            print("Test -- ", str(test_score), '\tTime={:.2f}s'.format(end_time - start_time),
                                  flush=True)

                if (batch_idx + 1) % self.check_every == 0 or batch_idx + 1 == len(batches) :
                    eval()

            end_time = time.time()
            print(colored("Epoch ", 'yellow'), iteration, ": Obj=", all_loss, '\tTime=', end_time - start_time, flush=True)

        print("Best Result:", self.best_score)


    def lrDecay(self, trainer, epoch):
        lr = NetworkConfig.NEURAL_LEARNING_RATE / (1 + NetworkConfig.lr_decay * (epoch - 1))
        for param_group in trainer.param_groups:
            param_group['lr'] = lr
        print('learning rate is set to: ', lr)
        return trainer

    def learn(self, train_insts, max_iterations, dev_insts, test_insts, optimizer_str = 'adam', batch_size=1):

        if optimizer_str == "lbfgs":
            self.learn_lbfgs(train_insts, max_iterations, dev_insts)
            return

        insts_before_split = train_insts

        insts = self.split_instances_for_train(insts_before_split)
        self.all_instances = insts

        self.touch(insts)

        self.lock_it()

        parameters = filter(lambda p: p.requires_grad, self.parameters())

        self.fm.get_param_g().print_transition(self.compiler.labels)

        if optimizer_str == 'adam':
            optimizer = torch.optim.Adam(parameters)
        elif optimizer_str == 'sgd':
            print(colored("Using SGD: lr is: {}, L2 regularization is: {}".format(NetworkConfig.NEURAL_LEARNING_RATE,
                                                                                  NetworkConfig.l2), 'yellow'))
            optimizer = torch.optim.SGD(parameters, lr=NetworkConfig.NEURAL_LEARNING_RATE,
                                        weight_decay=NetworkConfig.l2)
        else:
            print(colored('Unsupported optimizer:', 'red'), optimizer_str)
            return

        self.best_score = None
        self.test_score = None

        if self.check_every == None:
            self.check_every = len(self.all_instances)



        print('Start Training...', flush=True)
        for iteration in range(max_iterations):
            all_loss = 0
            start_time = time.time()
            k = 0
            idx = 0

            for i in np.random.permutation(len(self.all_instances)):
                self.train()
                inst = self.all_instances[i]
                if inst.get_instance_id() > 0:
                    if k == 0:
                        optimizer.zero_grad()
                        self.zero_grad()
                    gold_network = self.get_network(i)
                    partition_network = self.get_network(i + 1)

                    if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
                        gold_network.touch()
                        partition_network.touch()

                    gold_network.nn_output = self.fm.build_nn_graph(inst)
                    partition_network.nn_output = gold_network.nn_output

                    label_score = self.forward(gold_network)
                    unlabel_score = self.forward(partition_network)
                    loss = -unlabel_score - label_score
                    all_loss += loss.item()

                    loss.backward()
                    idx += 1
                    k += 1
                    if k == batch_size or idx == len(self.all_instances) // 2:
                        k = 0
                        optimizer.step()

                    del label_score
                    del unlabel_score
                    del loss

                    if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
                        del gold_network
                        self.networks[i] = None
                        del partition_network
                        self.networks[i + 1] = None
                if NetworkConfig.ECHO_TRAINING_PROGRESS > 0 and (i + 1) % NetworkConfig.ECHO_TRAINING_PROGRESS == 0:
                    print('x', end='', flush=True)

                def eval():
                    start_time = time.time()
                    results = self.decode(dev_insts)
                    score = self.evaluator.eval(dev_insts)
                    end_time = time.time()
                    print("Dev  -- ", str(score), '\tTime={:.2f}s'.format(end_time - start_time), flush=True)

                    if self.best_score == None or score.larger_than(self.best_score):

                        if self.best_score == None:
                            self.best_score = score
                        else:
                            self.best_score.update_score(score)
                        self.save()

                        start_time = time.time()

                        results = self.decode(test_insts)

                        self.test_score = self.evaluator.eval(test_insts)
                        end_time = time.time()
                        print(colored("Test -- ", 'red'), str(self.test_score),
                              '\tTime={:.2f}s'.format(end_time - start_time), flush=True)
                        with open('result.txt', 'w') as f:
                            for inst in results:
                                f.write(str(inst.get_input()) +'\n')
                                f.write(str(inst.get_output()) +'\n')
                                f.write(str(inst.get_prediction()) +'\n')
                                f.write('\n')


                    else:
                        if NetworkConfig.ECHO_TEST_RESULT_DURING_EVAL_ON_DEV:
                            start_time = time.time()
                            self.decode(test_insts)
                            test_score = self.evaluator.eval(test_insts)
                            end_time = time.time()
                            print("Test -- ", str(test_score), '\tTime={:.2f}s'.format(end_time - start_time),
                                  flush=True)

                if self.check_every >= 0 and (idx + 1) % self.check_every == 0:
                    eval()

            if self.check_every >= 0:
                eval()

            end_time = time.time()
            print(colored("Iteration ", 'yellow'), iteration, ": Obj=", all_loss, '\tTime={:.2f}s'.format(end_time - start_time), flush=True)
            print()
            #torch.cuda.empty_cache()/
        self.fm.get_param_g().print_transition(self.compiler.labels)
        if self.check_every == -1: #Just take one evaluation at the end of training
            eval()

        print("Best Dev Result:", self.best_score)
        print("Best Test Result:", self.test_score)


    def learn_lbfgs(self, train_insts, max_iterations, dev_insts):


        insts_before_split = train_insts

        insts = self.split_instances_for_train(insts_before_split)
        self.all_instances = insts

        self.touch(insts)
        self.lock_it()

        optimizer = torch.optim.LBFGS(self.parameters())
        self.iteration = 0
        self.best_ret = [0, 0, 0]


        self.iteration = 0
        def closure():
            self.train()
            self.zero_grad()
            optimizer.zero_grad()

            all_loss = 0

            start_time = time.time()
            for i in range(len(self.all_instances)):
                inst = self.all_instances[i]
                if inst.get_instance_id() > 0:
                    network = self.get_network(i)
                    negative_network = self.get_network(i + 1)
                    network.nn_output = self.fm.build_nn_graph(inst)
                    negative_network.nn_output = network.nn_output


            for i in range(len(self.all_instances)):
                loss = self.forward(self.get_network(i))
                all_loss -= loss



            all_loss.backward()
            end_time = time.time()

            print(colored("Iteration ", 'yellow'), self.iteration, ": Obj=", all_loss.item(), '\tTime=', end_time - start_time, flush=True)

            start_time = time.time()
            self.decode(dev_insts)
            ret = self.evaluator.eval(dev_insts)
            end_time = time.time()
            print("Prec.: {0:.2f} Rec.: {1:.2f} F1.: {2:.2f}".format(ret[0], ret[1], ret[2]), '\tTime={:.2f}s'.format(end_time - start_time), flush=True)


            if self.best_ret[2] < ret[2]:
                self.best_ret = ret
                self.save()

            self.iteration += 1
            if self.iteration >= max_iterations:
                return 0

            return all_loss


        while self.iteration < max_iterations:
            optimizer.step(closure)


        print("Best Result:", self.best_ret)

    def forward(self, network):
        return network.inside()

    def get_network(self, network_id):

        if self.networks[network_id] != None:
            return self.networks[network_id]

        inst = self.all_instances[network_id]

        network = self.compiler.compile(network_id, inst, self.fm)
        self.networks[network_id] = network

        return network


    def touch(self, insts):
        print('Touching ...', flush=True)
        if self.networks == None:
            self.networks = [None for i in range(len(insts))]

        if NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
            self.fm.gnp.set_network2nodeid2nn_size(len(insts))

        if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
            print('Exit Full Touch ...')
            return


        # if NetworkConfig.IGNORE_TRANSITION:
        #     print('Ignore Transition...')
        #     return


        start_time = time.time()

        num_thread = NetworkConfig.NUM_THREADS

        if num_thread > 1:
            print('Multi-thread Touching...')
            num_networks = len(insts)
            num_per_bucket = num_networks // num_thread if num_networks % num_thread == 0 else num_networks // num_thread + 1


            def touch_networks(bucket_id):
                end = num_per_bucket * (bucket_id + 1)
                end = min(num_networks, end)
                counter = 1
                for network_id in range(num_per_bucket * bucket_id, end):
                    if counter % 100 == 0:
                        print('.', end='', flush=True)
                    network = self.get_network(network_id)
                    network.touch()
                    counter += 1

                    if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
                        del network


            processes = []
            for thread_idx in range(num_thread):
                p = Process(target=touch_networks(thread_idx))
                processes.append(p)
                p.start()

            for thread_idx in range(num_thread):
                processes[thread_idx].join()

        else:
            for network_id in range(len(insts)):
                if network_id % 100 == 0:
                    print('.', end='', flush=True)
                network = self.get_network(network_id)

                network.touch()

                if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
                    del network

        end_time = time.time()

        print(flush=True)
        print('Toucing Completes taking ', end_time - start_time, ' seconds.', flush=True)

        if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
            del self.networks
            self.networks = [None] * len(insts)


    def touch_batch(self, insts, batches, batch_size, is_train=True):
        print('Touching ...', flush=True)
        # if self.networks == None:
        #     self.networks = [None for i in range(len(insts))]

        if NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
            self.fm.gnp.set_network2nodeid2nn_batch_size(len(batches))

        # if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
        #     print('Exit Full Touch ...')
        #     return


        # if NetworkConfig.IGNORE_TRANSITION:
        #     print('Ignore Transition...')
        #     return


        start_time = time.time()

        for batch_id in range(len(batches)):
            if batch_id % 100 == 0:
                print('.', end='', flush=True)

            for item in batches[batch_id]:
                item.touch(is_train)
            # batches[batch_id][0].touch()
            # batches[batch_id][1].touch()

            # if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
            #     del network

        end_time = time.time()

        print(flush=True)
        print('Toucing Completes taking ', end_time - start_time, ' seconds.', flush=True)

        # if not NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH:
        #     del self.networks
        #     self.networks = [None] * len(insts)




    def test(self, instances):
        return self.decode(instances=instances)

    def test_batch(self, instances, batch_size):
        return self.decode_batch(instances=instances, batch_size=batch_size)

    def decode(self, instances, cache_features=False):

        self.all_instances_test = instances
        self.eval()
        instances_output = []

        for k in range(len(instances)):
            instance = instances[k]

            network = self.compiler.compile(k, instance, self.fm)
            network.touch(is_train = False)
            network.nn_output = self.fm.build_nn_graph(instance)
            network.max()
            instance_output = self.compiler.decompile(network)
            instances_output.append(instance_output)

        return instances_output


    def decode_batch(self, instances, batch_size = 10):

        self.all_instances_test = instances

        batches = self.fm.generate_batches(self.all_instances_test, batch_size)

        batch_tensor_networks = []

        for batch_idx, batch in enumerate(batches):
            batch_input_seqs, batch_network_id_range = batch
            inst_ids = list(range(batch_network_id_range[0], batch_network_id_range[1]))

            batch_unlabel_networks = [self.compiler.compile(i, self.all_instances_test[i], self.fm) for i in inst_ids]

            batch_tensor_unlabel_networks = BatchTensorNetwork(self.fm, batch_idx, batch_unlabel_networks, batch_network_id_range)

            batch_tensor_networks.append((batch_tensor_unlabel_networks))

        # self.touch_batch(self.all_instances_test, batch_tensor_networks, batch_size, is_train=False)

        self.eval()
        instances_output = []

        for batch_idx, batch in enumerate(batches):
            batch_input_seqs, batch_network_id_range = batch
            batch_tensor_network = batch_tensor_networks[batch_idx]
            batch_tensor_network.touch(is_train = False)
            batch_tensor_network.nn_batch_output = self.fm.build_nn_graph_batch(batch_input_seqs)
            batch_tensor_network.max()

            for nid, network in enumerate(batch_tensor_network.batch_networks):
                network.max_paths = batch_tensor_network.max_paths[nid]
                instance_output = self.compiler.decompile(network)
                instances_output.append(instance_output)

        return instances_output


        # for k in range(len(instances)):
        #     instance = instances[k]
        #
        #     #network = self.compiler.compile(k, instance, self.fm)
        #     network.touch(is_train = False)
        #     network.nn_output = self.fm.build_nn_graph(instance)
        #     network.max()
        #     instance_output = self.compiler.decompile(network)
        #     instances_output.append(instance_output)

        # return instances_output


    # def get_network_test(self, network_id):
    #     if self.networks_test[network_id] != None:
    #         return self.networks_test[network_id]
    #
    #     inst = self.all_instances_test[network_id]
    #
    #     network = self.compiler.compile(network_id, inst, self.fm)
    #
    #
    #     self.networks_test[network_id] = network
    #
    #     return network


    # def touch_test(self, insts):
    #     if self.networks_test == None:
    #         self.networks_test = [None for i in range(len(insts))]
    #
    #     for network_id in range(len(insts)):
    #         if network_id % 100 == 0:
    #             print('.', end='')
    #         network = self.get_network_test(network_id)
    #
    #         network.touch()
    #
    #     print()


    def save(self):
        torch.save(self.state_dict(), self.model_path)
        print(colored('Save the best model to ', 'red'), self.model_path)

    def load(self):
        print(colored('Load the best model from ', 'red'), self.model_path)
        self.load_state_dict(torch.load(self.model_path))


    def set_visualizer(self, visualizer):
        self.visualizer = visualizer


    def visualize(self, network_id):
        network = self.get_network(network_id)
        self.visualizer.visualize(network)