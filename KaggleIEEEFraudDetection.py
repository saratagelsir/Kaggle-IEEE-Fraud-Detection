import os
import sys
import shutil
import traceback
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

CodeDirectory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CodeDirectory)
from ModelUtils import *
from EnsembleRusBoosting import EnsembleRUSBoosting


class KaggleIEEEFraudDetection:
    def __init__(self):
        # Initialize model parameters
        self.model_start = time.time()
        self.model_name = self.__class__.__name__
        self.user = get_username()
        self.model_run = ''
        self.run_datetime = pd.to_datetime('today')
        self.is_generate_report = False
        self.is_save_model_output = False
        self.randstate = 0

        self.response = u'isFraud'
        self.predictors = []
        self.interpolation_method = ''
        self.model_dir = get_current_dir()
        self.key_features = [u'TransactionID']
        self.feature_properties = pd.DataFrame()
        self.trained_model_file = ''
        self.null_cutoff = 0.3
        self.score_cutoff = 0.5

        self.idx_train = []
        self.idx_test = []
        self.data = pd.DataFrame()
        self.model = type('', (), {})()
        self.categorical_handler = type('', (), {})()

        log_writer('User %s started %s at %s' % (self.user, self.model_name, time.strftime('%Y-%m-%d %H:%M:%S')))
        log_writer('The model directory is: $MODELDIR=%s' % self.model_dir)
        log_writer('An empty object from the class %s is created' % self.model_name)

    def parse_model_info(self, model_run):
        # Parse model_run string
        self.model_run = model_run

        # Model files
        data_folder = os.path.join(self.model_dir, 'data')
        filename = os.path.join(data_folder, 'feature_properties.xlsx')
        self.feature_properties = pd.read_excel(filename)
        self.trained_model_file = os.path.join(data_folder, 'fraud_detection_trained_model.pkl')
        self.model = EnsembleRUSBoosting(base_estimator='tree', n_estimators=1000, learning_rate=0.2, with_replacement=False,
                                         random_state=self.randstate)
        log_writer('Finished parsing model information')

    def read_data_from_files(self):
        filename = lambda x: os.path.join(self.model_dir, 'data', '%s_%s.csv' % (self.model_run, x))
        identity = pd.read_csv(filename('identity'))
        transaction = pd.read_csv(filename('transaction'))
        data = pd.merge(identity, transaction, on=self.key_features, how='outer')
        data = data.sort_values(self.key_features).reset_index(drop=True)

        # Ordered columns
        idx = self.feature_properties[u'Features'].isin(data.columns)
        ordered_cols = self.feature_properties.loc[idx, u'new_Features'].tolist()
        data = data[self.feature_properties.loc[idx, u'Features'].tolist()].copy(deep=True)
        data.columns = ordered_cols

        self.data = data.copy(deep=True)
        self.feature_selection()

    def feature_selection(self):
        if self.model_run == 'test':
            # Load trained pot leakage model and predict
            self.model, self.categorical_handler = joblib.load(self.trained_model_file)
            self.predictors = self.model.features
            return

        flag = 2
        null_percs = self.data.isnull().sum().values / float(self.data.shape[0])
        if flag == 1:
            # Select columns with null percentage < 30%
            idx = null_percs < self.null_cutoff
        else:
            # Choose columns with k most less null percentages
            idx = null_percs.argsort()[:102]

        cols = self.data.columns.values[idx].tolist()

        if self.response in cols:
            cols.remove(self.response)

        for feature in self.key_features:
            cols.remove(feature)

        # Choose the most non null columns
        self.predictors = cols

    def pick_features_subset(self, column, label):
        idx_cols = self.feature_properties[column].isin([label])
        sub_features = self.feature_properties.loc[idx_cols, u'new_Features']
        return sub_features[sub_features.isin(self.data.columns.values)].values

    def data_imputation_all_columns(self):
        interpolator_column = u'Imputation'
        interpolators = self.feature_properties[interpolator_column].unique()

        for interpolator in interpolators:
            # Get features for the interpolator
            interpolator_features = self.pick_features_subset(interpolator_column, interpolator)
            if len(interpolator_features) == 0:
                continue

            if interpolator == 'linear':
                # Linear interpolation
                self.data[interpolator_features] = self.data[interpolator_features].interpolate(method='linear')

            elif interpolator == 'fillprevious':
                # Backward fill
                self.data[interpolator_features] = self.data[interpolator_features].ffill().bfill()

            elif interpolator == 'fillzeros':
                # Fill Na/NaN with zeros
                self.data[interpolator_features] = self.data[interpolator_features].fillna(0)

            elif interpolator == 'filldummy':
                # For categorical features fill empty with 'XX'
                func = lambda x: x.replace('', np.nan).fillna('X' * x.fillna('').astype(str).str.len().max())
                self.data[interpolator_features] = self.data[interpolator_features].apply(func)

            elif interpolator == 'nofill':
                # Ignore for nofill
                continue

            else:
                # Unknown method
                log_writer('Interpolator/imputation method: ''%s'' is unknown or not implemented' % interpolator)
                continue

        log_writer('Finished data imputation for all features/tags')

    def training_testing_data_splitting(self):
        log_writer('Splitting model data into training/testing datasets and training the model')

        # Model output initialization
        self.data[u'ydataM'] = False
        self.data[u'score_proba'] = 0.0

        # Divide data to training and validation datasets.
        if self.model_run == 'train':
            self.data[self.response].astype(bool)
            idx_fraud = np.where(self.data[self.response])[0]
            n = int(idx_fraud.shape[0] * 0.7)
            last_train_obs = idx_fraud[n]
            self.idx_train = self.data.index <= last_train_obs
            self.idx_test = self.data.index > last_train_obs
        else:
            self.idx_train = np.array([False] * self.data.shape[0])
            self.idx_test = np.array([True] * self.data.shape[0])

        self.data[u'idx_train'] = self.idx_train
        self.data[u'idx_test'] = self.idx_test

        # Print training/testing splitting stats
        line = u'-' * (32 + 1)
        print('%s\n___________Training______Testing_' % line)
        print('# Obs. %12d %12d' % tuple(self.data[[u'idx_train', u'idx_test']].sum()))
        if self.model_run == 'train':
            true_count = lambda x: self.data.loc[x, self.response].isin([1]).sum()
            false_count = lambda x: self.data.loc[x, self.response].isin([0]).sum()
            print('# 0\'s  %12d %12d' % (false_count(self.idx_train), false_count(self.idx_test)))
            print('# 1\'s  %12d %12d' % (true_count(self.idx_train), true_count(self.idx_test)))

        print(line)

    def compute_performance_stats(self):
        if self.model_run == 'test':
            return

        cm_train = confusion_matrix(self.data.loc[self.idx_train, self.response], self.data.loc[self.idx_train, u'ydataM'], labels=[False, True])
        cm_test = confusion_matrix(self.data.loc[self.idx_test, self.response], self.data.loc[self.idx_test, u'ydataM'], labels=[False, True])
        print('Training Confusion Matrix:')
        print(cm_train.T)
        print('Testing Confusion Matrix:')
        print(cm_test.T)

        # Calculate AUC
        auc_train = roc_auc_score(self.data.loc[self.idx_train, self.response], self.data.loc[self.idx_train, u'score_proba'])
        print('Training AUC: %.3f' % auc_train)

        auc_test = roc_auc_score(self.data.loc[self.idx_test, self.response], self.data.loc[self.idx_test, u'score_proba'])
        print('Testing AUC:  %.3f' % auc_test)

        # Calculate roc curve
        fpr_train, tpr_train, thrshlds_train = roc_curve(self.data.loc[self.idx_train, self.response], self.data.loc[self.idx_train, u'score_proba'])
        fpr_test, tpr_test, thrshlds_test = roc_curve(self.data.loc[self.idx_test, self.response], self.data.loc[self.idx_test, u'score_proba'])
        # Plot the roc curve for the model
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr_train, tpr_train, marker='*', label='Train')
        plt.plot(fpr_test, tpr_test, marker='.', label='Test')
        plt.legend()
        filename = os.path.join(self.model_dir, 'data', 'trained_model_auc.png')
        plt.savefig(filename, bbox_inches='tight')

    def fraud_detection_model_train(self):
        # Training and testing data
        X_train = self.data.loc[self.idx_train, self.predictors].reset_index(drop=True)
        X_test = self.data.loc[self.idx_test, self.predictors].reset_index(drop=True)

        # Categorical features
        self.categorical_handler = OrdinalEncoder()
        self.categorical_handler = self.categorical_handler.fit(X_train)

        X_train = self.categorical_handler.transform(X_train)
        X_test = self.categorical_handler.transform(X_test)
        y_train = self.data.loc[self.idx_train, self.response].reset_index(drop=True)

        # Train the model
        self.model.fit(X_train, y_train)

        score_train = self.model.predict_proba(X_train)
        score_test = self.model.predict_proba(X_test)

        self.data.loc[self.idx_train, u'score_proba'] = score_train[:, 1]
        self.data.loc[self.idx_test, u'score_proba'] = score_test[:, 1]

        self.data.loc[self.idx_train, u'ydataM'] = (score_train[:, 1] >= self.score_cutoff)
        self.data.loc[self.idx_test, u'ydataM'] = (score_test[:, 1] >= self.score_cutoff)

        log_writer('Saving the calibrated machine learning model in: %s for later use' % print_path(self.trained_model_file))
        if os.path.isfile(self.trained_model_file):
            shutil.move(self.trained_model_file, os.path.splitext(self.trained_model_file)[0] + '_old.pkl')

        joblib.dump([self.model, self.categorical_handler], self.trained_model_file)

    def fraud_detection_model_test(self):
        X_test = self.data.loc[self.idx_test, self.predictors].reset_index(drop=True)

        # Categorical features
        X_test = self.categorical_handler.transform(X_test)

        # Apply the model
        score_test = self.model.predict_proba(X_test)
        self.data.loc[self.idx_test, u'score_proba'] = score_test[:, 1]
        self.data.loc[self.idx_test, u'ydataM'] = (score_test[:, 1] >= self.score_cutoff)

    def fraud_detection_main_method(self, model_run):
        status = 1
        description = ''
        model_output = ''
        self.randstate = 0
        np.random.seed(self.randstate)

        try:
            self.parse_model_info(model_run)
            self.read_data_from_files()
            self.data_imputation_all_columns()
            # TODO: check this
            self.data[self.predictors] = self.data[self.predictors].fillna(self.data.median())
            self.training_testing_data_splitting()
            model_target_func = getattr(self, 'fraud_detection_model_%s' % self.model_run)
            model_target_func()
            self.compute_performance_stats()
        except Exception:
            status = -1
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err_list = traceback.format_exception(exc_type, exc_value, exc_traceback)
            err_log = clean_string(print_path(''.join(err_list)))
            description = 'Failure: ' + err_log

        if status == 1:
            description = 'Model run finished successfully at %s (runtime %.2f secs)' % (
                time.strftime('%Y-%m-%d %H:%M:%S'), (time.time() - self.model_start))
        else:
            model_output = '{"status": %d, "description": "%s"}' % (status, description)

        log_writer(description)
        return model_output


if __name__ == '__main__':
    modelrun = sys.argv[1] if len(sys.argv) > 1 else 'train'
    fd = KaggleIEEEFraudDetection()
    model_output = fd.fraud_detection_main_method(modelrun)
