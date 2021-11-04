import sys
import numpy as np
import os
import pickle
import pandas as pd
import itertools

from collections import Counter
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler as resample

lables_path = './labels/'
features_path = './features/'
results_path = './modelling_results/'

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

import tensorflow_addons as tfa
METRICS = [
	  tf.keras.metrics.TruePositives(name='tp'),
	  tf.keras.metrics.FalsePositives(name='fp'),
	  tf.keras.metrics.TrueNegatives(name='tn'),
	  tf.keras.metrics.FalseNegatives(name='fn'),
	  tf.keras.metrics.BinaryAccuracy(name='accuracy'),
	  tf.keras.metrics.Precision(name='precision'),
	  tf.keras.metrics.Recall(name='recall'),
	  tf.keras.metrics.AUC(name='auc'),
	  tf.keras.metrics.BinaryCrossentropy(name='BinaryCrossentropy')
]

class KerasWorker(Worker):
	def __init__(self, input_shape, output_shape, problemType,
				 x_train, y_train, x_validation, y_validation,
				 x_test, y_test, shared_directory, **kwargs):
			super().__init__(**kwargs)
			self.input_shape = (input_shape, )
			self.num_classes = output_shape
			self.batch_size = 64
			self.save_dic = shared_directory
			
			self.problemType = problemType

			self.x_train, self.y_train = x_train, y_train
			self.x_validation, self.y_validation = x_validation, y_validation
			self.x_test, self.y_test = x_test, y_test

	def compute(self, config, budget, working_directory, *args, **kwargs):
			model = Sequential()
			model.add(Dense(units=config['start_neurons_units'],
							# activation=config['start_neurons_activation'],
							activation='relu',
							input_shape=self.input_shape))


			if config['num_dense_layers'] > 1:
				model.add(Dense(units=config['dense1_units'],
								# activation=config['dense1_activation'],
								activation='relu',
								input_shape=self.input_shape))
				model.add(Dropout(config['dropout1_rate']))

			if config['num_dense_layers'] > 2:
				model.add(Dense(units=config['dense2_units'],
								# activation=config['dense2_activation'],
								activation='relu',
								input_shape=self.input_shape))
				model.add(Dropout(config['dropout2_rate']))

			model.add(Dense(self.num_classes, activation='softmax'))


			if config['optimizer'] == 'Adam':
					optimizer = tf.keras.optimizers.Adam(lr=config['lr'])
			else:
					optimizer = tf.keras.optimizers.SGD(lr=config['lr'], momentum=config['sgd_momentum'])
			
			if self.problemType == 'classification':
				loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
				the_metrics = METRICS.append(tfa.metrics.MatthewsCorrelationCoefficient(num_classes=self.num_classes))
				the_columns = ['loss','tp','fp','tn','fn','acc','prec','rec','auc','BC','MCC']
				#val_metric = 'accuracy'
				val_metric = 'val_loss'
			else:
				loss_fn = tf.keras.losses.MeanSquaredError()
				the_metrics=['mean_squared_error', 'mean_absolute_error', 
							 'mean_absolute_percentage_error', 'cosine_proximity',
							'mean_squared_logarithmic_error']
				the_columns = the_metrics
				val_metric = 'mean_squared_error'
				
			model.compile(
				loss=loss_fn,
				optimizer=optimizer,
				metrics=the_metrics
			)

			# model.summary()
			_history = model.fit(self.x_train, self.y_train,
							  batch_size=self.batch_size,
							  epochs=int(budget),
							  verbose=0,
							  validation_data=(self.x_validation, self.y_validation))

			print(_history.history.keys())
			val_acc_per_epoch = _history.history[val_metric]
			best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1 \
						 if self.problemType == 'classification' \
						 else val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
			print('Best epoch: %d' % (best_epoch,))

			model.fit(self.x_train, self.y_train,
							  batch_size=self.batch_size,
							  epochs=best_epoch,
							  verbose=0,
							  validation_data=(self.x_validation, self.y_validation))

			train_score = model.evaluate(self.x_train, self.y_train, verbose=0)
			val_score = model.evaluate(self.x_validation, self.y_validation, verbose=0)
			test_score = model.evaluate(self.x_test, self.y_test, verbose=0)

			resultsDF = pd.DataFrame([train_score,val_score,test_score],
									 columns=the_metrics,
									 index=["train_score", "val_score", "test_score"],)
			# print(resultsDF)
			test_predictions_baseline = model.predict(self.x_test)
			np.savetxt(os.path.join(self.save_dic,'testing_finalResults_true.out'), self.y_test, delimiter=',')
			np.savetxt(os.path.join(self.save_dic,'testing_finalResults_pred.out'), test_predictions_baseline, delimiter=',')

			return ({
				'loss': test_score,  
				'info':  resultsDF.to_dict('index')
			})

	@staticmethod
	def get_configspace():
			"""
			It builds the configuration space with the needed hyperparameters.
			It is easily possible to implement different types of hyperparameters.
			Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
			:return: ConfigurationsSpace-Object
			"""
			cs = CS.ConfigurationSpace()

			lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

			# For demonstration purposes, we add different optimizers as categorical hyperparameters.
			# To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
			# SGD has a different parameter 'momentum'.
			optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

			sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

			cs.add_hyperparameters([lr, optimizer, sgd_momentum])



			num_dense_layers =  CSH.UniformIntegerHyperparameter('num_dense_layers', lower=1, upper=3, default_value=2)

			start_neurons_units = CSH.UniformIntegerHyperparameter('start_neurons_units', lower=32, upper=512, default_value=32, log=True)
			dense1_units = CSH.UniformIntegerHyperparameter('dense1_units', lower=8, upper=128, default_value=16, log=True)
			dense2_units = CSH.UniformIntegerHyperparameter('dense2_units', lower=4, upper=64, default_value=8, log=True)

			cs.add_hyperparameters([num_dense_layers, start_neurons_units, dense1_units, dense2_units])

			# start_neurons_activation = CSH.CategoricalHyperparameter('start_neurons_activation', ['relu', 'tanh', 'sigmoid'])
			# dense1_activation = CSH.CategoricalHyperparameter('dense1_activation', ['relu', 'tanh', 'sigmoid'])
			# dense2_activation = CSH.CategoricalHyperparameter('dense2_activation', ['relu', 'tanh', 'sigmoid'])
			# start_neurons_activation = CSH.CategoricalHyperparameter('start_neurons_activation', ['relu'])
			# dense1_activation = CSH.CategoricalHyperparameter('dense1_activation', ['relu'])
			# dense2_activation = CSH.CategoricalHyperparameter('dense2_activation', ['relu'])
			#
			# cs.add_hyperparameters([start_neurons_activation, dense1_activation, dense2_activation])

			dropout1_rate = CSH.UniformFloatHyperparameter('dropout1_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
			dropout2_rate = CSH.UniformFloatHyperparameter('dropout2_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)

			cs.add_hyperparameters([dropout1_rate, dropout2_rate])


			# The hyperparameter sgd_momentum will be used,if the configuration
			# contains 'SGD' as optimizer.
			cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
			cs.add_condition(cond)

			# You can also use inequality conditions:
			cond = CS.GreaterThanCondition(dense1_units, num_dense_layers, 1)
			cs.add_condition(cond)

			cond = CS.GreaterThanCondition(dense2_units, num_dense_layers, 2)
			cs.add_condition(cond)

			return cs

def load_sub_dataset_twostream(familiesSet, label_folder, feature_folder):
	# append all rows of subjects, and their lables
	allFrames = np.array([])
	allLables = np.array([])
	
	rgb_feature_folder = feature_folder.replace(featureType,'rgb')
	flow_feature_folder = feature_folder.replace(featureType,'flow')
	for this_family in familiesSet:
		onlyfiles = [f for f in os.listdir(rgb_feature_folder) if
					   os.path.isfile(os.path.join(rgb_feature_folder, f))
					   and f.startswith(this_family + '_')]
		onlyfiles.sort()

		for this_file in onlyfiles:
			this_lable_file = this_file.replace('_'+this_file.split('_')[4],'')
			currLabel = np.load(os.path.join(label_folder,this_lable_file))
			
			rgb_currData = np.load(os.path.join(rgb_feature_folder,this_file))	
			flow_currData = np.load(os.path.join(flow_feature_folder,this_file.replace('rgb','flow')))
			
			currData = np.hstack((rgb_currData, flow_currData))
			
			if allFrames.shape[0] ==0:
				allFrames = currData
				allLables = currLabel
			else:
				allFrames = np.vstack((allFrames, currData))
				allLables = np.hstack((allLables, currLabel))

	return allFrames, allLables

def load_sub_dataset(familiesSet, label_folder, feature_folder, featureType):
	# append all rows of subjects, and their lables
	allFrames = np.array([])
	allLables = np.array([])
	
	if featureType == 'twostream':
		return load_sub_dataset_twostream(familiesSet, label_folder, feature_folder)
	
	for this_family in familiesSet:
		# F10_Interaction_1_P27_rgb.npy
		onlyfiles = [f for f in os.listdir(feature_folder) if
					   os.path.isfile(os.path.join(feature_folder, f))
					   and f.startswith(this_family + '_')]
		onlyfiles.sort()

		for this_file in onlyfiles:
			currData = np.load(os.path.join(feature_folder,this_file))
			this_lable_file = this_file.replace('_'+this_file.split('_')[4],'')+'.npy'
			currLabel = np.load(os.path.join(label_folder,this_lable_file))
			
			if allFrames.shape[0] ==0:
				allFrames = currData
				allLables = currLabel
			else:
				allFrames = np.vstack((allFrames, currData))
				allLables = np.hstack((allLables, currLabel))

	return allFrames, allLables

def load_dataset_selectedSubj(trainSubjs, valSubjs, testSubjs, label_folder, feature_folder,\
							  prblemType, featureType, num_classes):
	#simple sampeling method
	#TODO: SMOTE, DeepSMOTE, DeepFake?
	sm = resample()
	# load all train
	trainX, trainy = load_sub_dataset(trainSubjs, label_folder, feature_folder, featureType)
	trainX, trainy = shuffle(trainX, trainy)
	trainX, trainy = sm.fit_resample(trainX, trainy)
	print(trainX.shape, Counter(trainy))
	# train_class_weight = sumary_data(trainy)

	# load validation
	valX, valy = load_sub_dataset(valSubjs, label_folder, feature_folder, featureType)
	valX, valy = shuffle(valX, valy)
	valX, valy = sm.fit_resample(valX, valy)
	print(valX.shape, Counter(valy))
	# val_class_weight = sumary_data(valy)


	# load all test
	testX, testy = load_sub_dataset(testSubjs, label_folder, feature_folder, featureType)
	testX, testy = sm.fit_resample(testX, testy)
	print(testX.shape, Counter(testy))
	# test_class_weight = sumary_data(testy)

	# one hot encode y
	if prblemType == 'classification':
		trainy = tf.keras.utils.to_categorical(trainy,  num_classes=num_classes)
		valy = tf.keras.utils.to_categorical(valy,  num_classes=num_classes)
		testy = tf.keras.utils.to_categorical(testy,  num_classes=num_classes)
	
	return trainX, trainy, valX, valy , testX, testy, num_classes
	
def create_modlling(label_folder,feature_folder,result_folder, prblemType, featureType, num_classes):	
	# repeat experiment
	temp = {}
	all_trainSubjs = [['F' + str(i) for i in [1, 2, 3, 4, 5, 6, 8]]]
	all_valSubjs = [['F' + str(i) for i in [11, 17]]]
	all_testSubjs = [['F' + str(i) for i in [7, 10, 13]]]
	
	min_budget = 9
	max_budget = 243
	n_iterations = 50
	num_workers = 12

	for r in range(len(all_trainSubjs)):
		shared_directory = result_folder + '_'+ str(r) 
		if os.path.exists(shared_directory):
			print(shared_directory,' already processed')
			continue
		
		print(shared_directory,' under processing')
		classType = os.path.basename(shared_directory)

		host = hpns.nic_name_to_host('lo')
		result_logger = hpres.json_result_logger(directory=shared_directory, overwrite=True)
		NS = hpns.NameServer(run_id=classType, host=host, port=0, working_directory=shared_directory)
		ns_host, ns_port = NS.start()
	
		# load data
		trainSubjs = all_trainSubjs[r]
		valSubjs = all_valSubjs[r]
		testSubjs = all_testSubjs[r]
		
		trainX, trainy, valX, valy, testX, testy, num_classes = \
			load_dataset_selectedSubj(trainSubjs, valSubjs, testSubjs, \
									  label_folder, feature_folder, prblemType, featureType, num_classes)
		
		
		n_timesteps, n_features, n_outputs = trainX.shape[0], trainX.shape[1], num_classes		
		
		workers = []
		for i in range(num_workers):
			worker = KerasWorker(n_features, n_outputs, prblemType, \
								 trainX, trainy, valX, valy, testX, testy, \
								 shared_directory,
								 run_id=classType,host=host, nameserver=ns_host, nameserver_port=ns_port,
								 id=i)
			worker.run(background=True)
			workers.append(worker)

		bohb = BOHB(configspace=worker.get_configspace(),
				  run_id=classType,
				  host=host,
				  nameserver=ns_host,
				  nameserver_port=ns_port,
				  result_logger=result_logger,
				  min_budget=min_budget, max_budget=max_budget
					)
		res = bohb.run(n_iterations=1,  min_n_workers=num_workers)

		id2config = res.get_id2config_mapping()
		incumbent = res.get_incumbent_id()

		print('Best found configuration:', id2config[incumbent]['config'])
		# print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
		# print('A total of %i runs where executed.' % len(res.get_all_runs()))
		# print('Total budget corresponds to %.1f full function evaluations.' % (
		#			 sum([r.budget for r in res.get_all_runs()]) / max_budget))

		# store results
		with open(os.path.join(shared_directory, 'results.pkl'), 'wb') as fh:
			pickle.dump(res, fh)

		# shutdown
		bohb.shutdown(shutdown_workers=True)
		NS.shutdown()

if __name__ == "__main__":
	prblemTypes = ['classification', 'regression']
	featureTypes = ['rgb','flow', 'twostream']
	classes = [9,5,3]
	# divides = [2.5, 5]
	fusionTypes = ['none', 'conc','avg']

	permutations=[ prblemTypes, featureTypes, classes, fusionTypes]
	all_permutations = list(itertools.product(*permutations))
	print(len(all_permutations))
	for this_permutation in all_permutations:
		(prblemType, featureType, eng_lvls, fusionType) = this_permutation
		classType = 'round_avg_eng_level' if prblemType == 'classification' else 'avg_eng_level'
		divide = 2.5 if fusionType == 'none' else 5
		
		print('Working on: ',prblemType, featureType, eng_lvls, divide, fusionType)
		
		label_folder = os.path.join(lables_path,'_'.join([classType,'eng_lvl',prblemType,str(eng_lvls),str(divide)]))

		
		if divide == 5:
			extra_txt = '_'.join([fusionType,str(divide)]) 
			feature_folder = os.path.join(features_path,'_'.join(['i3d',featureType,'features',extra_txt]))
		else:
			feature_folder = os.path.join(features_path,'_'.join(['i3d',featureType,'features']))
			
		
		#save results like: rgb_classification_9_2.5_none
		result_folder = os.path.join(results_path,'_'.join([featureType,prblemType,str(eng_lvls),str(divide),fusionType]))

		create_modlling(label_folder,feature_folder,result_folder, prblemType, featureType, eng_lvls)#, divide,fusionType)