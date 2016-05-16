""" Author: Gokhan Cetinkaya, gcastorrr@gmail.com

    This repo contains several experimental classes which randomly generate and evolve features (concepts) for supervised classification problems.
    As mentioned, code is not ready/intended for production use.
    Documentation will be provided in the future.

"""

import numpy as np
import pandas as pd
import time
import copy
import pylab
from sklearn.metrics import roc_auc_score as auc


def softmax(l):
  """ Returns softmax values for given array.
  """
  ex = np.exp(l)
  return ex/np.sum(ex, axis=0)


class BaseEvolver(object):
  """ Abstract class for different kinds of evolvers.
  """

  def __init__(self, n_concepts=10, p_mut=0.01, p_co=0.75, n_gens=50, pop_size=100, hof=1, random_state=time.time(), co_type=None, mate_selection="roulette"):
    """ params:
          n_concepts      : number of concepts to be learned.
          p_mut           : probability of mutation.
          p_co            : probability of cross-over.
          n_gens          : number of generations.
          pop_size        : number of individuals in population.
          hof             : number of best individuals to persist at hall of fame.
          random_state    : seed np.random to have the ability to use same randomness.

    """
    if n_concepts < 1 or not isinstance(n_concepts, int):
      raise Exception("Invalid n_concepts. Must be int > 0")

    self.n_concepts = n_concepts
    self.p_mut = p_mut
    self.p_co = p_co
    self.n_gens = n_gens
    self.pop_size = pop_size
    self.hof = hof
    self.hof_ind_score = [(0,0) for _ in range(self.hof)] if self.hof else [(0,0)]
    np.random.seed(int(random_state))
    self.co_type = co_type
    self.mate_selection = mate_selection

    self.population = []
    self.gen_history = []
    self.best_individual = []
    self._best_individuals = []
    self.best_individual_test_scores = []
    self.last_gen = 0
    self.ignore_eval_features_gen = []

  def _init_population(self):
    """ Initializes a random population
    """
    self.population = []
    self.gen_history = []
    self.last_gen = 0
    for i in range(self.pop_size):
      self.population.append(self._get_random_individual())

  def _eval_gen(self, generation):
      """ Evaluates a given generation.
          First, computes scores of each individual. Then, performs cross-over and mutation operations on the population and generates a new population.
          Persists history records.
      """
      ind_scores = []
      data = self.data_folds[generation % self.n_folds]
      labels = self.label_folds[generation % self.n_folds].values.ravel().astype(int)
      for i in range(self.pop_size):
        ind_scores.append( self._eval_individual(self.population[i], data, labels) )

      try:
        self.gen_history[generation] = ind_scores
        self._best_individuals[generation] = self.population[np.argmax(ind_scores)]
      except Exception:
        self.gen_history.append( ind_scores )
        self._best_individuals.append( self.population[np.argmax(ind_scores)] )
      self.best_individual = self.population[np.argmax(ind_scores)]
      # persist best_fold_individual
      best_score = np.max(ind_scores)
      current_best_fold_ind_score = self._best_fold_individuals[generation%self.n_folds]["score"]
      if best_score > current_best_fold_ind_score:
        self._best_fold_individuals[generation%self.n_folds] = {"score": best_score, "ind": self.best_individual}

      if self.n_folds > 1:
        test_fold = np.random.choice(range(1, self.n_folds), size=1)
        data_test = self.data_folds[generation % test_fold]
        labels_test = self.label_folds[generation % test_fold].values.ravel().astype(int)
        try:
          self.best_individual_test_scores[generation] = self._eval_individual(self.best_individual, data_test, labels_test)
        except IndexError:
          self.best_individual_test_scores.append(self._eval_individual(self.best_individual, data_test, labels_test))

      if self.eval_features_every_n_gens and generation not in self.ignore_eval_features_gen and generation%self.eval_features_every_n_gens == 0:
        if self._eval_features():
          self._eliminate_poor_features()

      return ind_scores

  def _persist_hof(self, scores):
    """ Persists best individual to Hall of Fame (hof).
    """
    if not self.hof:
      return

    scores = copy.copy(scores)
    for i in range(self.hof):
      ind = np.argmax(scores)
      self.hof_ind_score[i] = (self.population[ind], scores[ind]) #(individual, score)
      scores.pop()

  def _add_hof_to_pop(self):
    """ Adds individuals in hof to population.
    """
    i = 0
    for ind, score in self.hof_ind_score:
      self.population[i] = ind
      i += 1

  def _get_mates(self, scores_proba):
    """ Returns 2 individuals from given scores_proba based on mate_selection param.
    """
    mate_indexes = [-1,-1]
    if self.mate_selection == "roulette":
      mate_indexes = np.random.choice(range(len(scores_proba)), size=2, replace=False, p=scores_proba)
    elif self.mate_selection == "tournament":
      mate_candidates = np.random.choice(range(len(scores_proba)), size=4, replace=False)
      mate_indexes[0] = mate_candidates[0] if scores_proba[mate_candidates[0]] >= scores_proba[mate_candidates[1]] else mate_candidates[1]
      mate_indexes[1] = mate_candidates[2] if scores_proba[mate_candidates[2]] >= scores_proba[mate_candidates[3]] else mate_candidates[3]
    else:
      raise Exception("invalid mate_selection param")

    return mate_indexes

  def _get_offsprings(self, new_pop, scores_proba):
    """ Returns an offspring from current population using cross-over operation.
    """
    mate_indexes = self._get_mates(scores_proba) #np.random.choice(range(len(scores_proba)), size=2, replace=False, p=scores_proba)

    if np.random.rand() < self.p_co: # crossover
      self._crossover(new_pop, mate_indexes)
    else: # no crossover
      new_pop.append(self.population[mate_indexes[0]])
      new_pop.append(self.population[mate_indexes[1]])

  def _crossover(self, new_pop, mate_indexes):
    """ Performs (2-point) crossover operation and persists generated offsprings at new_pop.
    """
    ind1 = self.population[mate_indexes[0]]
    ind2 = self.population[mate_indexes[1]]

    if self.co_type == "cx_mult_genes":
      off1, off2 = self._cx_mult_genes(ind1, ind2)
    else:
      cxp_indexes = np.random.choice(range(self._individual_length), size=2, replace=False)
      cxp_indexes.sort()
      off1 = np.concatenate( [ ind1[0:cxp_indexes[0]], ind2[cxp_indexes[0]:cxp_indexes[1]], ind1[cxp_indexes[1]:] ], axis=0)
      off2 = np.concatenate( [ ind2[0:cxp_indexes[0]], ind1[cxp_indexes[0]:cxp_indexes[1]], ind2[cxp_indexes[1]:] ], axis=0)

    if len(off1) != self._individual_length or len(off2) != self._individual_length:
      raise Exception("Invalid offsprings:\noff1:%s\noff2:%s" % (off1, off2))

    new_pop.append(off1)
    new_pop.append(off2)

  def _mate_gen(self, scores_proba):
    """ Creates the new generation based on current population (w scores) and hof with cross-over.
    """
    new_pop = []
    while len(new_pop) < self.pop_size:
      self._get_offsprings(new_pop, scores_proba)

    self.population = new_pop

  def _get_mean_score(self, n_last_gens=None):
    """ Returns mean score of n_last_gens.
        Useful for deciding the overall score of a call to fit() w n_folds.
    """
    if not n_last_gens:
      n_last_gens = self.n_folds
    gen_scores = self.gen_history[-self.n_folds:]
    gen_scores = [np.max(x) for x in gen_scores]

    return np.mean(gen_scores)

  def _get_max_score(self):
    """ Returns mean score of n_last_gens.
        Useful for deciding the overall score of a call to fit() w n_folds.
    """
    max_scores = [ np.max(x) for x in self.gen_history ]

    return np.max(max_scores)

  def fit(self, X, y, n_classes, n_folds=10, warm_start=False, verbose=1):
    """ Processes evolution for given X and y.
    """
    if not warm_start:
      self._init_data(X, y, n_classes, n_folds)
      self._init_population()
      start_gen = 0
      end_gen = self.n_gens
    else:
      start_gen = self.last_gen + 1
      end_gen = start_gen + self.n_gens

    # do NOT eval_features for start_gen and end_gen..
    self.ignore_eval_features_gen.append(start_gen)
    self.ignore_eval_features_gen.append(end_gen)
    for i in range(start_gen, end_gen):
      scores = self._eval_gen(i)
      scores_proba = [x/float(sum(scores)) for x in scores]
      self._persist_hof(scores) # store hof -do not alter population
      self._mate_gen(scores_proba) # from now on, self.population is the new generation!
      self._mutate_gen()
      self._add_hof_to_pop()
      self.last_gen = i
      if verbose:
        print "Gen %04d Best: %.4f (ix:%03d) Mean: %.4f Std: %.4f" % (i, max(self.gen_history[i]), np.argmax(self.gen_history[i]), np.asarray(self.gen_history[i]).mean(),  np.asarray(self.gen_history[i]).std())

    self._eval_gen(i+1)
    if verbose:
      print "Gen %04d Best: %.4f (ix:%03d) Mean: %.4f Std: %.4f" % (i+1, max(self.gen_history[i+1]), np.argmax(self.gen_history[i+1]), np.asarray(self.gen_history[i+1]).mean(),  np.asarray(self.gen_history[i+1]).std())
      print "***** BEST INDIVIDUAL ******:\n%s" % self.best_individual
      print "Mean Score: %.4f" % self._get_mean_score()
      print "Max Score: %.4f" % self._get_max_score()


class RandomConceptEvolver2(BaseEvolver):
  """ A Generic Classifier based on Random Bit Regression article.
      n_attrs_per_feature   : Determines how many attributes will be used for a feature. Can be a fixed int or list of tuples (ie. [ (n_attrs_candidate_1, proba1), (n_attrs_candidate_2, proba2), ... ])

  """
  def __init__(self, n_concepts=10, p_mut=0.05, p_mut_gene=0.1, p_co=0.75, n_gens=50, pop_size=100, hof=1,
                  random_state=time.time(), co_type=None, mate_selection="tournament", n_attrs_per_feature=3):

    if co_type == "cx_mult_genes":
      raise Exception("RandomConceptEvolver2 cannot use cx_mult_genes crossover")
    BaseEvolver.__init__(self, n_concepts=n_concepts, p_mut=p_mut, p_co=p_co, n_gens=n_gens, pop_size=pop_size, hof=hof,
                            random_state=random_state, co_type=co_type, mate_selection=mate_selection)
    self.p_mut_gene = p_mut_gene
    self.n_attrs_per_feature = n_attrs_per_feature

  def _get_random_indexes_weights(self):
    """ Generates a random feature.
        Determines number of attributes to use by self.n_attrs_per_feature variable.
    """
    if type(self.n_attrs_per_feature) == int:
      n_attrs_for_feat = self.n_attrs_per_feature
    else:
      n_attrs_candidates = [ nattr for nattr, prob in self.n_attrs_per_feature ]
      attr_probas = [ prob for nattr, prob in self.n_attrs_per_feature ]
      n_attrs_for_feat = np.random.choice(n_attrs_candidates, size=1, p=attr_probas, replace=False)

    attr_indexes = np.random.choice(range(self.n_attrs), size=n_attrs_for_feat, replace=False)
    weights = np.array( [ np.random.normal() for _ in range(n_attrs_for_feat) ] ) # weights are selected from normal distribution

    return attr_indexes, weights

  def _get_random_feature(self, data):
    """ Generates a random feature and populates it's values by given data.
    """
    while True:
      attr_indexes, weights = self._get_random_indexes_weights()
      concept_data = data[:,attr_indexes]
      features = np.dot(concept_data, weights)
      unique_features = np.unique(features)
      # if it's a binary feature, set threshold accordingly..
      if len(unique_features) == 2:
        threshold = unique_features[1]
      else:
        threshold = np.random.choice(features)
      features[features>=threshold] = 1
      features[features<threshold] = 0
      # if there's no variance in created feature, discard it..
      if features.std() == 0.0:
        continue

      return attr_indexes, weights, threshold, features

  def _generate_features(self, data):
    """ Generates random bits as intermediate features for given data.
    """
    print "Generating feature matrix..."
    data = data.as_matrix() # back to numpy
    self._feature_schema = []
    self._features = np.zeros( [self.n_samples, self.n_concepts] )
    for c in range(self.n_concepts):
      attr_indexes, weights, threshold, feature = self._get_random_feature(data)
      self._features[:,c] = feature
      self._feature_schema.append(((attr_indexes), (weights), threshold))
      # if c%int(self.n_concepts/100) == 0:
      #   print(".", end="")

    print "done"

  def _init_data(self, data, y, n_classes, n_folds):
    """ Initializes data fields to be used during evolution.
        params:
          data            : dataset to be used during evolution. pandas.DataFrame is expected.
          y               : labels
          n_classes       : number of possible classes.
          n_folds         : number of folds to split the data. In every generation, only one fold is used. if n_folds == 1, all data is used in every generation.
    """
    if data.empty:
      raise Exception("data is required")
    if not n_classes:
      raise Exception("n_classes is required")
    if n_folds < 1 or type(n_folds) != int:
      raise Exception("Invalid n_folds. Must be integer > 0")

    try:
      self.n_samples = data.shape[0]
      self.n_attrs = data.shape[1]
      self.data = data
      self.labels = y
      self.n_folds = n_folds

    except Exception as e:
      raise Exception("Invalid data format. Expected a pandas.DataFrame.")

    if len(y) != self.n_samples:
      raise Exception("X and y must have equal length")

    # generate features matrix (random bits)
    self._generate_features(data)

    self.fold_size = self.n_samples / n_folds
    self.data_folds = []
    self.label_folds = []
    self._best_fold_individuals = {}
    for i in range(n_folds):
      if i == n_folds - 1: # if last fold, take all samples till the end
        self.data_folds.append(self._features[i * self.fold_size : ])
        self.label_folds.append(self.labels[i * self.fold_size : ])
      else:
        self.data_folds.append(self._features[i * self.fold_size : (i+1) * self.fold_size])
        self.label_folds.append(self.labels[i * self.fold_size : (i+1) * self.fold_size])
      self._best_fold_individuals[i] = {"score": 0.0, "ind": np.array([])}

    self.n_classes = n_classes
    self._concept_length = 1 # only initialized for compatibility
    self._individual_length = self.n_concepts * self._concept_length

  def _get_random_individual(self):
    """ Generates a random individual.
        An individual is simply an array of classes (eg. [<class1>, <class2>] ) for all n_concepts.
        If data (which is binary) has value 1, and the corresponding gene of the individual is "2", the individual has 1 vote for class "2".
        Value 0 indicates a NO VOTE situation.
    """
    return np.random.choice(range(self.n_classes+1), size=self._individual_length)

  def _get_votes(self, ind, data):
    """ Returns votes for ind and data.
    """
    # print "ind:\n%s" % ind
    # print "data:\n%s" % data
    ind_ext = np.array([ ind, ]*data.shape[0]) # extends individual for all samples.
    # print "ind_ext:\n%s" % ind_ext

    # print "votes:\n%s" % (data * ind_ext)
    return data * ind_ext

  def _get_predictions(self, ind, data):
    """ Returns predictions of given individual for given data.
    """
    votes = self._get_votes(ind, data)
    # print "votes:\n%s" % votes
    preds = []
    for sample in range(data.shape[0]):
      unique, counts = np.unique(votes[sample,:], return_counts=True)
      unique = unique[1:] # remove 0
      counts = counts[1:] # remove 0 count
      # print "unique:\n%s" % unique
      # print "counts:\n%s" % counts
      try:
        preds.append(unique[np.argmax(counts)]-1) # make the pred index(1-based) as in same format as labels(0-based). (We'd added 0 for "no vote")
      except ValueError:
        raise ValueError("Not enough concepts for data")

    return np.array(preds)

  def _eval_individual(self, ind, data, labels):
    """ Evaluates and scores a given individual, by given data and labels.
    """
    preds = self._get_predictions(ind, data)
    if self.eval_metric == "auc":
      perf = auc(labels, preds) # auc
    else: #acc
      perf = np.sum(preds==labels)/float(len(labels)) # acc

    return perf**2

  def _mutate_gen(self):
    """ Mutates current population w given p_mut.
        Does NOT care about concepts linked to the same attr more than once!
    """
    n_inds_to_mutate = int(self.pop_size*self.p_mut)
    if n_inds_to_mutate:
      inds_to_mutate_indexes = np.random.choice(range(self.pop_size), size=n_inds_to_mutate, replace=False)

    for ix in inds_to_mutate_indexes:
      ind = self.population[ix]
      n_genes_to_mutate = int(self._individual_length*self.p_mut_gene)
      n_genes_to_mutate = 1 if n_genes_to_mutate < 1 else n_genes_to_mutate
      mut_indexes = np.random.choice(range(self._individual_length), size=n_genes_to_mutate, replace=False)
      for i in mut_indexes:
        ind[i] = np.random.choice(range(self.n_classes+1), size=1)[0]

  def _add_hof_to_pop(self):
    """ Adds best individuals to population.
    """
    best_individuals = self._best_individuals[-self.n_folds:]
    i = 0
    for ind in best_individuals:
      self.population[i] = ind
      i += 1

  def _get_feature(self, data, concept_ix):
    """ Generates and returns feature values for given data and concept index by _feature_schema.
    """
    attr_indexes, weights, threshold = self._feature_schema[concept_ix]
    concept_data = data[:,attr_indexes]
    features = np.dot(concept_data, weights)
    features[features>=threshold] = 1
    features[features<threshold] = 0

    return features

  def _get_features_from_schema(self, data):
    """ Generates and returns features from schema for given data.
    """
    if not self._feature_schema:
      raise Exception("no valid schema")

    features = np.zeros( [data.shape[0], self.n_concepts] )
    data = data.as_matrix() # back to numpy
    for c in range(self.n_concepts):
      feature = self._get_feature(data, c)
      features[:,c] = feature

    return features

  def predict(self, data):
    """ Returns predictions based on current state of the model for given data.
    """
    if len(self.best_individual) == 0:
      raise Exception("No current model (best_individual)")

    _features = self._get_features_from_schema(data)

    preds = np.zeros([data.shape[0], self.n_folds])
    for f in range(self.n_folds):
      preds[:,f] = self._get_predictions(self._best_fold_individuals[f]["ind"], _features)

    final_preds = []
    for sample in range(data.shape[0]):
      unique, counts = np.unique(preds[sample,:], return_counts=True)
      # unique = unique[1:] # remove 0
      # counts = counts[1:] # remove 0 count
      # print "unique:\n%s" % unique
      # print "counts:\n%s" % counts
      try:
        final_preds.append(unique[np.argmax(counts)]-1) # make the pred index(1-based) as in same format as labels(0-based). (We'd added 0 for "no vote")
      except ValueError:
        raise ValueError("Not enough concepts for predicting data")

    return np.array(final_preds)

  def plot(self):
    """ Plots max, mean and std accuracy levels by generation.
    """
    max_scores = [ np.sqrt(np.max(x)) for x in self.gen_history ]
    mean_scores = [ np.sqrt(np.mean(x)) for x in self.gen_history ]
    std_scores = [ np.sqrt(np.std(x)) for x in self.gen_history ]
    max_vald_scores = [ np.sqrt(x) for x in self.best_individual_test_scores ]

    plt_max_scores, = pylab.plot(range(len(self.gen_history)), max_scores, label="Max")
    plt_mean_scores, = pylab.plot(range(len(self.gen_history)), mean_scores, label="Mean")
    plt_std_scores, = pylab.plot(range(len(self.gen_history)), std_scores, label="Std Deviation")
    plt_vald_scores, = pylab.plot(range(len(self.gen_history)), max_vald_scores, label="Validation")
    pylab.title("Performance")
    pylab.legend(handles=[plt_max_scores, plt_mean_scores, plt_std_scores, plt_vald_scores], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pylab.xlabel("n_gens")
    pylab.ylabel("Accuracy")
    pylab.show()

class RandomConceptEvolver3(RandomConceptEvolver2):
  """ New Evolver based on RandomConceptEvolver2.
      Added functionalities:
        1) The environment params (generated features) are also evolved.
        2)
  """
  def __init__(self, n_concepts=10, p_mut=0.05, p_mut_gene=0.1, p_co=0.75, n_gens=50, pop_size=100, hof=1,
                  random_state=time.time(), co_type=None, mate_selection="tournament", n_attrs_per_feature=2,
                  eval_features_every_n_gens=100, eliminate_perc_features=0.05, eval_metric="acc"):
    if eval_metric not in ["acc", "auc"]:
      raise Exception("Invalid eval_metric. Expected acc or acu.")

    RandomConceptEvolver2.__init__(self, n_concepts=n_concepts, p_mut=p_mut, p_mut_gene=p_mut_gene, p_co=p_co, n_gens=n_gens, pop_size=pop_size, hof=hof,
                            random_state=random_state, co_type=co_type, mate_selection=mate_selection, n_attrs_per_feature=n_attrs_per_feature)
    self.eval_features_every_n_gens = eval_features_every_n_gens
    self.eliminate_perc_features = eliminate_perc_features
    self.eval_metric = eval_metric

  def _eval_features(self):
    """ Evaluates features by evaluating best_individual(s) when a feature is/not present.
    """
    if not self.eval_features_every_n_gens:
      return None

    self._feature_scores = []
    best_ind_scores = np.array([ np.max(scores) for scores in self.gen_history[-self.n_folds:] ])

    # best_ind_scores = []
    # for ind in self._best_individuals[-self.n_folds:]:
    #   # print "self._features.shape: %s" % str(self._features.shape)
    #   # print "self.labels.shape: %s" % str(self.labels.shape)
    #   best_ind_scores.append( self._eval_individual(ind, self._features, self.labels.values.ravel().astype(int)) )
    # best_ind_scores = np.array(best_ind_scores)
    last_generation_index = len(self.gen_history)-1
    for c in range(self.n_concepts):
      best_ind_scores_wout_concept = []
      rewind_generation = self.n_folds
      for ind in self._best_individuals[-self.n_folds:]:
        gen_index = last_generation_index - rewind_generation
        data = self.data_folds[gen_index % self.n_folds]
        labels = self.label_folds[gen_index % self.n_folds].values.ravel().astype(int)
        features_wout_concept = copy.copy(data)
        features_wout_concept[:,c] = np.zeros(features_wout_concept.shape[0])
        best_ind_scores_wout_concept.append( self._eval_individual(ind, features_wout_concept, labels) )
        rewind_generation -= 1

      best_ind_scores_wout_concept = np.array( best_ind_scores_wout_concept )
      self._feature_scores.append( (c, np.sum( best_ind_scores_wout_concept - best_ind_scores ) ) )

    print "finished evaluating features"

    return True

  def _eliminate_feature(self, c):
    """ Removes given feature from _features and updates _feature_schema.
        Does not alter population.
    """
    attr_indexes, weights, threshold, feature = self._get_random_feature(self.data.as_matrix())
    self._features[:,c] = feature
    self._feature_schema[c] = ((attr_indexes), (weights), threshold)

  def _re_init_data_folds(self):
    """ Reinitializes data_folds to reflect changes in features.
    """
    self.data_folds = []
    for i in range(self.n_folds):
      if i == self.n_folds - 1: # if last fold, take all samples till the end
        self.data_folds.append(self._features[i * self.fold_size : ])
      else:
        self.data_folds.append(self._features[i * self.fold_size : (i+1) * self.fold_size])

  def _eliminate_poor_features(self):
    """ Eliminates bad performing features.
    """
    print "started feature elimination process..."
    feat_scores_sorted = sorted( self._feature_scores, key=lambda tup: tup[1] )
    # find features beyond 90th percentile
    feat_only_scores = np.array([ x[1] for x in feat_scores_sorted ])
    feat_scores_percentile = feat_only_scores[feat_only_scores > np.percentile(feat_only_scores, 90)]
    n_feat_scores_percentile = len(feat_scores_percentile)
    max_n_features_to_eliminate = int(self.n_concepts * self.eliminate_perc_features)
    n_features_to_eliminate = min(n_feat_scores_percentile, max_n_features_to_eliminate)
    if not n_features_to_eliminate:
      print "no feature to eliminate"
      return
    feature_indexes_to_eliminate = [ x[0] for x in feat_scores_sorted[-n_features_to_eliminate:] ]
    for feat_ix in feature_indexes_to_eliminate:
      self._eliminate_feature(feat_ix)

    print "eliminated %d features:" % n_features_to_eliminate
    for f,s in feat_scores_sorted[-n_features_to_eliminate:]:
      print "%d: %f" % (f,s)

    self._re_init_data_folds()



class RandomConceptEvolver(BaseEvolver):
  """ Evolves random concepts for classification.
      An individual has:
        int(n_attrs * n_attrs_per_concept_ratio) bits per concept and
        a total of (n_classes * n_concepts_per_class) * n_concepts bits.

  """

  def __init__(self, n_attrs_per_concept_ratio=0.05, n_concepts_per_class=10, p_mut=0.01, p_co=0.75, n_gens=50, pop_size=100, hof=1, random_state=time.time(), co_type=None):
    """ params:
          n_attrs_per_concept_ratio : number of attrs per concept as a ratio of total number of attrs.(ie. a total of 770 attrs & n_attrs_per_concept_ratio=0.05 yields ~38 attrs per concept)
          n_concepts_per_class      : number of concepts to be learned per class.
          p_mut                     : probability of mutation.
          p_co                      : probability of cross-over.
          n_gens                    : number of generations.
          pop_size                  : number of individuals in population.
          hof                       : number of best individuals to persist at hall of fame.
          random_state              : seed np.random to have the ability to use same randomness.

    """
    if n_concepts_per_class < 1 or not isinstance(n_concepts_per_class, int):
      raise Exception("Invalid n_concepts. Must be int > 0")

    self.n_concepts_per_class = n_concepts_per_class
    self.n_attrs_per_concept_ratio = n_attrs_per_concept_ratio
    self.p_mut = p_mut
    self.p_co = p_co
    self.n_gens = n_gens
    self.pop_size = pop_size
    self.hof = hof
    self.hof_ind_score = [(0,0) for _ in range(self.hof)] if self.hof else [(0,0)]
    np.random.seed(int(random_state))
    self.co_type = co_type

    self.population = []
    self.gen_history = []
    self.last_gen = 0

  def _init_data(self, data, y, n_classes, n_folds):
    """ Initializes data fields to be used during evolution.
        params:
          data            : dataset to be used during evolution. pandas.DataFrame is expected.
          y               : labels
          n_classes       : number of possible classes.
          n_folds         : number of folds to split the data. In every generation, only one fold is used. if n_folds == 1, all data is used in every generation.
    """
    if data.empty:
      raise Exception("data is required")
    if not n_classes:
      raise Exception("n_classes is required")
    if n_folds < 1 or type(n_folds) != int:
      raise Exception("Invalid n_folds. Must be integer > 0")

    try:
      self.n_samples = data.shape[0]
      self.n_attrs = data.shape[1]
      self.data = data
      self.labels = y
      self.n_folds = n_folds

    except Exception as e:
      raise Exception("Invalid data format. Expected a pandas DataFrame.")

    if len(y) != self.n_samples:
      raise Exception("X and y must have equal length")

    self.fold_size = self.n_samples / n_folds
    self.data_folds = []
    self.label_folds = []
    for i in range(n_folds):
      if i == n_folds - 1: # if last fold, take all samples till the end
        self.data_folds.append(self.data[i * self.fold_size : ])
        self.label_folds.append(self.labels[i * self.fold_size : ])
      else:
        self.data_folds.append(self.data[i * self.fold_size : (i+1) * self.fold_size])
        self.label_folds.append(self.labels[i * self.fold_size : (i+1) * self.fold_size])

    self.n_classes = n_classes
    self._concept_length = int(self.n_attrs_per_concept_ratio * self.n_attrs)
    self.n_concepts = self.n_concepts_per_class * self.n_classes
    self._individual_length = self._concept_length * self.n_concepts

  def _get_random_individual(self):
    """ Generates a random individual.
        An individual is simply an array of (<attr_index>, <weight>) tuples.
        _individual's array structure is as follows:
          [ (attr1_index, weight1), (attr2_index, weight2), ... ]
    """
    weights = np.random.randn(self._individual_length)
    attr_indexes = np.zeros(self._individual_length)
    for i in range(self.n_concepts):
      attr_indexes[i*self._concept_length:(i+1)*self._concept_length] = np.random.choice(range(self.n_attrs), size=self._concept_length, replace=False)
    #attr_indexes = np.asarray(attr_indexes)

    return np.asarray(zip(attr_indexes, weights))

  def _get_class_outputs(self, ind, X):
    """ Returns class scores for X (only 1 sample) for given individual.
        NOTE: We can enhance this method to support multiple samples in which case we must use np.einsum for efficiency.
    """
    class_outputs = {}
    concepts = ind.reshape(self.n_concepts, self._concept_length, 2)
    i = 0
    for concept in concepts:
      cls_index = i%self.n_classes

      # print "shape of np.array(concept[:,1]): %s" % np.array(concept[:,1]).shape
      # print "shape of np.array(concept[:,1]): %s" % np.array(concept[:,1]).shape

      output = np.dot([X[0,x] for x in concept[:,0]], concept[:,1])
      if class_outputs.get(cls_index):
        class_outputs[cls_index].append(output)
      else:
        class_outputs[cls_index] = [output]
      i += 1

    return class_outputs

  def _get_pred_eff_score(self, class_outputs, correct_cls_index):
    """ Calculates a heuristic efficiency score for given class_outputs (of individual) for the given correct label.
        A correct prediction has a coefficient of +1 and an incorrect prediction (class_output) has a coeff of -1.
    """
    eff_score = 0
    for cls_index, outputs in class_outputs.items():
      if cls_index == correct_cls_index: # correct label
        for output in outputs:
          if output > 0: # correctly predicted +
            eff_score += output
          else:
            eff_score += -output
      else: # incorrect label
        for output in outputs:
          if output > 0: # incorrectly predicted +
            eff_score += -output
          else:
            eff_score += output

    return eff_score

  def _eval_individual(self, ind, data, labels):
    """ Evaluates and scores a given individual, by given data and labels.
    """
    eff_scores = []
    for sample in range(len(data)):
      #X = data[sample:sample+1].as_matrix()
      X = data[sample:sample+1].as_matrix()
      class_outputs = self._get_class_outputs(ind, X)
      eff_scores.append( self._get_pred_eff_score(class_outputs, labels[sample]) )

    # remove negative scores
    eff_scores = [x if x > 0 else 0.0000000001 for x in eff_scores]

    return np.mean(eff_scores)

  def _cx_mult_genes(self, ind1, ind2):
    """ Performs 2-point crossover across multiple genes (concepts) of indiduals.
        A gene is never split into multiple parts during this kind of crossover.
    """
    if self.n_concepts < 3:
      raise Exception("n_concepts must be greater than 2")

    cxp_indexes = np.random.choice(range(self.n_concepts), size=2, replace=False)*self._concept_length
    cxp_indexes.sort()
    off1 = np.concatenate( [ ind1[0:cxp_indexes[0]], ind2[cxp_indexes[0]:cxp_indexes[1]], ind1[cxp_indexes[1]:] ], axis=0)
    off2 = np.concatenate( [ ind2[0:cxp_indexes[0]], ind1[cxp_indexes[0]:cxp_indexes[1]], ind2[cxp_indexes[1]:] ], axis=0)

    return off1, off2

  def _mutate_gen(self):
    """ Mutates current population w given p_mut.
        Does NOT care about concepts linked to the same attr more than once!
        Mutation probability of attr_index and weights is equal (ie. 50% - 50%)
    """
    for ind in self.population:
      mut_indexes = [x for x in range(self._individual_length) if np.random.rand() < self.p_mut]
      for i in mut_indexes:
        if np.random.rand() < 0.5: #mutate attr_index
          ind[i][0] = np.random.choice(range(self.n_attrs), size=1)[0]
        else: #mutate weight
          ind[i][1] = np.random.randn()

  def fit(self, X, y, n_classes, n_folds=10, warm_start=False, verbose=1):
    """ Processes evolution for given X and y.
    """
    if not warm_start:
      self._init_data(X, y, n_classes, n_folds)
      self._init_population()
      start_gen = 0
      end_gen = self.n_gens

    else:

      start_gen = self.last_gen + 1
      end_gen = start_gen + self.n_gens

    for i in range(start_gen, end_gen):
      scores = self._eval_gen(i)
      scores_proba = [x/float(sum(scores)) for x in scores]
      self._persist_hof(scores) # store hof -do not alter population
      self._mate_gen(scores_proba) # from now on, self.population is the new generation!
      self._mutate_gen()
      self._add_hof_to_pop()
      self.last_gen = i
      if verbose:
        print "Gen %04d Best: %.4f (ix:%03d) Mean: %.4f Std: %.4f" % (i, max(self.gen_history[i]), np.argmax(self.gen_history[i]), np.asarray(self.gen_history[i]).mean(),  np.asarray(self.gen_history[i]).std())

    self._eval_gen(i+1)
    if verbose:
      print "Gen %04d Best: %.4f (ix:%03d) Mean: %.4f Std: %.4f" % (i+1, max(self.gen_history[i+1]), np.argmax(self.gen_history[i+1]), np.asarray(self.gen_history[i+1]).mean(),  np.asarray(self.gen_history[i+1]).std())
      print "***** BEST INDIVIDUAL ******:\n%s" % self.best_individual


class ConceptEvolutionClassifier(BaseEvolver):
  """ Class for using evolution on classification problems by means of generating concepts.
      Does NOT auto scale data.
  """

  def __init__(self, n_concepts=10, p_mut=0.01, p_co=0.75, n_gens=50, pop_size=100, hof=1, random_state=time.time(), co_type=None):
    """ params:
          n_concepts      : number of concepts to be learned.
          p_mut           : probability of mutation.
          p_co            : probability of cross-over.
          n_gens          : number of generations.
          pop_size        : number of individuals in population.
          hof             : number of best individuals to persist at hall of fame.
          random_state    : seed np.random to have the ability to use same randomness.

    """
    if n_concepts < 1 or not isinstance(n_concepts, int):
      raise Exception("Invalid n_concepts. Must be int > 0")

    self.n_concepts = n_concepts
    self.p_mut = p_mut
    self.p_co = p_co
    self.n_gens = n_gens
    self.pop_size = pop_size
    self.hof = hof
    self.hof_ind_score = [(0,0) for _ in range(self.hof)] if self.hof else [(0,0)]
    np.random.seed(int(random_state))
    self.co_type = co_type

    self.population = []
    self.gen_history = []
    self.last_gen = 0

  def _init_data(self, data, y, n_classes, n_folds):
    """ Initializes data fields to be used during evolution.
        params:
          data            : dataset to be used during evolution. pandas.DataFrame is expected.
          y               : labels
          n_classes       : number of possible classes.
          n_folds         : number of folds to split the data. In every generation, only one fold is used. if n_folds == 1, all data is used in every generation.
    """
    if data.empty:
      raise Exception("data is required")
    if not n_classes:
      raise Exception("n_classes is required")
    if n_folds < 1 or type(n_folds) != int:
      raise Exception("Invalid n_folds. Must be integer > 0")

    try:
      self.n_samples = data.shape[0]
      self.n_attrs = data.shape[1]
      self.data = data
      self.labels = y
      self.n_folds = n_folds

    except Exception as e:
      raise Exception("Invalid data format. Expected a pandas DataFrame.")

    if len(y) != self.n_samples:
      raise Exception("X and y must have equal length")

    self.fold_size = self.n_samples / n_folds
    self.data_folds = []
    self.label_folds = []
    for i in range(n_folds):
      if i == n_folds - 1: # if last fold, take all samples till the end
        self.data_folds.append(self.data[i * self.fold_size : ])
        self.label_folds.append(self.labels[i * self.fold_size : ])
      else:
        self.data_folds.append(self.data[i * self.fold_size : (i+1) * self.fold_size])
        self.label_folds.append(self.labels[i * self.fold_size : (i+1) * self.fold_size])

    self.n_classes = n_classes
    self._concept_length = 1 + self.n_attrs + self.n_classes # +1 for bias unit
    self._individual_length = self.n_concepts * self._concept_length + self.n_classes # self.n_classes for class bias units

  def _get_random_individual(self):
    """ Generates a random individual.
        An individual is simply an array of incoming and outgoing weights for all concepts.
        _individual's array structure is as follows:
          [.........XXXXXXX..........XXXXXXXX............XXXXXXXXCCCC] where ... s represent incoming weights, XXXs represent outgoing weights and CCC represent class biases.
    """
    return np.random.randn(self._individual_length)


  def _get_ind_concept_weights(self, ind):
    """ Returns all weights for all concepts for given individual.
    """
    return [ ind[i*self._concept_length : (i+1)*self._concept_length ] for i in range(self.n_concepts)]

  def _get_ind_weights(self, ind):
    """ Decomposes an individual to its in and out weights.
        weights_in has length of n_concepts and every item has length of n_attrs +1. (ie. incoming weights for all concepts for all attrs)
        weights_out has length of n_concepts and every item has length of n_classes. (ie. outgoing weights for all concepts for all classes)
    """
    weights = self._get_ind_concept_weights(ind)

    # store in and out weights for each concept
    weights_in = [ x[ 0:self.n_attrs + 1 ] for x in weights]
    weights_out = [ x[ self.n_attrs + 1 : ] for x in weights]
    class_biases = ind[ -self.n_classes: ]

    return weights_in, weights_out, class_biases

  def _get_concept_activations(self, weights_in, X):
    """ Computes activations (outputs) of all concepts for given weights and sample data (X).
    """
    concept_activations = []
    for con in range(self.n_concepts):
      w_all = weights_in[con]
      bias = w_all[ 0 : 1 ]
      w = w_all[ 1 : ]
      # print "X:%s" % X
      # print "w_all:%s" % w_all
      # print "bias:%s" % bias
      # print "w:%s" % w
      # print "np.dot(X,w):%s" % np.dot(X,w)
      # print "sum(list(bias)):%s" % sum(list(bias))

      concept_activations.append( np.dot(X,w) + bias )

    return concept_activations

  def _get_class_activations(self, weights_out, class_biases, concept_outputs):
    """ Computes activations (outputs) of all classes (output layer) for given weights and concept_outputs.
    """
    class_activations = []
    for cla in range(self.n_classes):
      w = np.asarray([x[cla] for x in weights_out])
      bias = class_biases[ cla ]
      # print "w:%s" % w
      # print "bias:%s" % bias
      # print "concept_outputs:%s" % concept_outputs
      # print "np.dot(concept_outputs, w) + bias: %f" % (np.dot(concept_outputs, w) + bias)
      class_activations.append( np.dot(concept_outputs, w) + bias )

    return class_activations

  def _get_class_prediction(self, weights_in, weights_out, class_biases, X):
    """ Returns class predictions for given in/out weights and sample data.
        Utilizes softmax.
    """
    concept_activations = self._get_concept_activations(weights_in, X)
    # concept_outputs = np.asarray([np.tanh(x) for x in concept_activations])
    concept_outputs = [float(np.tanh(x)) for x in concept_activations]
    # print "weights_out: %s" % weights_out
    class_activations = self._get_class_activations(weights_out, class_biases, concept_outputs)
    # print "class_activations: %s" % class_activations

    return np.argmax(softmax(class_activations))

  def _eval_individual(self, ind, data, labels):
    """ Evaluates and scores a given individual, by given data and labels.
    """
    weights_in, weights_out, class_biases = self._get_ind_weights(ind)

    class_predictions = []
    for sample in range(len(data)):
      X = data[sample:sample+1].as_matrix()
      class_predictions.append( self._get_class_prediction(weights_in, weights_out, class_biases, X) )

    preds = np.asarray(class_predictions)
    #labels = np.asarray(labels)
    acc = np.sum(preds==labels)/float(len(labels))

    if acc < 0 or acc > 1:
      print "ind: %s" % ind
      print "class_predictions: %s" % class_predictions
      print "labels: %s" % labels
      print "np.sum(preds==labels)/float(len(labels)): %f" % (np.sum(preds==labels)/float(len(labels)))

    return acc**2 # return accuracy


  def _cx_mult_genes(self, ind1, ind2, cx_ix_concepts=None, cx_ix_classes=None):
    """ Performs 2-point crossover across multiple genes (concepts) of indiduals.
        Class biasses are additionally crossovered.
    """
    if self.n_concepts < 3:
      raise Exception("n_concepts must be greater than 2")

    if not cx_ix_concepts:
      cx_ix_concepts = np.random.choice(range(1,self.n_concepts), size=2, replace=False)
    cx_ix_concepts.sort()

    # if binary classification, do not cx class biases
    if self.n_classes > 2:
      if not cx_ix_classes:
        cx_ix_classes = np.random.choice(range(1,self.n_classes), size=2, replace=False)
      cx_ix_classes.sort()

    ind1_concepts = self._get_ind_concept_weights(ind1)
    ind1_class_biases = ind1[ -self.n_classes: ]
    ind2_concepts = self._get_ind_concept_weights(ind2)
    ind2_class_biases = ind2[ -self.n_classes: ]

    off1_concepts = np.concatenate( [ ind1_concepts[0:cx_ix_concepts[0]], ind2_concepts[cx_ix_concepts[0]:cx_ix_concepts[1]], ind1_concepts[cx_ix_concepts[1]:] ], axis=0 )
    if self.n_classes > 2:
      off1_class_biases = np.concatenate( [ ind1_class_biases[0:cx_ix_classes[0]], ind2_class_biases[cx_ix_classes[0]:cx_ix_classes[1]], ind1_class_biases[cx_ix_classes[1]:] ], axis=0 )
    else:
      off1_class_biases = ind1_class_biases
    off1 = np.concatenate ( [np.asarray(off1_concepts).flatten(), np.asarray(off1_class_biases).flatten()], axis=0 )

    off2_concepts = np.concatenate( [ ind2_concepts[0:cx_ix_concepts[0]], ind1_concepts[cx_ix_concepts[0]:cx_ix_concepts[1]], ind2_concepts[cx_ix_concepts[1]:] ], axis=0 )
    if self.n_classes > 2:
      off2_class_biases = np.concatenate( [ ind2_class_biases[0:cx_ix_classes[0]], ind1_class_biases[cx_ix_classes[0]:cx_ix_classes[1]], ind2_class_biases[cx_ix_classes[1]:] ], axis=0 )
    else:
      off2_class_biases = ind2_class_biases
    off2 = np.concatenate ( [np.asarray(off2_concepts).flatten(), np.asarray(off2_class_biases).flatten()], axis=0 )

    return off1, off2

  def _crossover(self, new_pop, mate_indexes):
    """ Performs (2-point) crossover operation and persists generated offsprings at new_pop.
    """
    ind1 = self.population[mate_indexes[0]]
    ind2 = self.population[mate_indexes[1]]

    if self.co_type == "cx_mult_genes":
      off1, off2 = self._cx_mult_genes(ind1, ind2)
    else:
      cxp_indexes = np.random.choice(range(self._individual_length), size=2, replace=False)
      cxp_indexes.sort()
      off1 = np.concatenate( [ ind1[0:cxp_indexes[0]], ind2[cxp_indexes[0]:cxp_indexes[1]], ind1[cxp_indexes[1]:] ], axis=0)
      off2 = np.concatenate( [ ind2[0:cxp_indexes[0]], ind1[cxp_indexes[0]:cxp_indexes[1]], ind2[cxp_indexes[1]:] ], axis=0)

    if len(off1) != self._individual_length or len(off2) != self._individual_length:
      raise Exception("Invalid offsprings:\noff1:%s\noff2:%s" % (off1, off2))

    new_pop.append(off1)
    new_pop.append(off2)


  def _mate_gen(self, scores_proba):
    """ Creates the new generation based on current population (w scores) and hof with cross-over.
    """
    new_pop = []
    while len(new_pop) < self.pop_size:
      self._get_offsprings(new_pop, scores_proba)

    # if pop_size is exceeded, remove the exceeding last individual
    if len(new_pop) > self.pop_size:
      new_pop = new_pop[:-1]

    self.population = new_pop

  def _mutate_gen(self):
    """ Mutates current population w given p_mut.
    """
    # for ind in self.population:
    #   mut_indexes = [x for x in range(self._individual_length) if np.random.rand() < self.p_mut]
    #   for i in mut_indexes:
    #     ind[i] = np.random.randn()
    inds_to_mutate = [x for x in range(self.pop_size) if np.random.rand() < self.p_mut]
    for i in inds_to_mutate:
      ind = self.population[i]
      bits_to_mutate = np.random.choice(range(self._individual_length), size=int(self.p_mut / 100.0 * self._individual_length))
      for mx in bits_to_mutate:
        ind[mx] = np.random.randn()



  def fit(self, X, y, n_classes, n_folds=10, warm_start=False, verbose=1):
    """ Processes evolution for given X and y.
    """
    if not warm_start:
      self._init_data(X, y, n_classes, n_folds)
      self._init_population()
      start_gen = 0
      end_gen = self.n_gens

    else:

      start_gen = self.last_gen + 1
      end_gen = start_gen + self.n_gens

    for i in range(start_gen, end_gen):
      scores = self._eval_gen(i)
      scores_proba = [x/float(sum(scores)) for x in scores]
      self._persist_hof(scores) # store hof -do not alter population
      self._mate_gen(scores_proba) # from now on, self.population is the new generation!
      self._mutate_gen()
      self._add_hof_to_pop()
      self.last_gen = i
      if verbose:
        print "Gen %04d Best: %.4f (ix:%03d) Mean: %.4f Std: %.4f" % (i, max(self.gen_history[i]), np.argmax(self.gen_history[i]), np.asarray(self.gen_history[i]).mean(),  np.asarray(self.gen_history[i]).std())

    self._eval_gen(i+1)
    if verbose:
      print "Gen %04d Best: %.4f (ix:%03d) Mean: %.4f Std: %.4f" % (i+1, max(self.gen_history[i+1]), np.argmax(self.gen_history[i+1]), np.asarray(self.gen_history[i+1]).mean(),  np.asarray(self.gen_history[i+1]).std())
      print "***** BEST INDIVIDUAL ******:\n%s" % self.best_individual

  def predict(self, data):
    """ Predict classes for given data(X) by using current best_individual.
    """
    if not self.best_individual:
      raise Exception("No current model (best_individual)")

    weights_in, weights_out, class_biases = self._get_ind_weights(self.best_individual)

    class_predictions = []
    for sample in range(len(data)):
      X = data[sample:sample+1].as_matrix()
      class_predictions.append( self._get_class_prediction(weights_in, weights_out, class_biases, X) )

    return np.asarray(class_predictions)



