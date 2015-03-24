import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                    filename="history.log", filemode='a', level=logging.DEBUG,
                                        datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                      datefmt='%m/%d/%y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logger = logging.getLogger(__name__)

def main(CONFIG):
    SEED = 314
    
    models = []
    model_dict = { 'LR'   : LogisticRegression,
                   'RFC'  : RandomForestClassifier,
                   'ETC'  : ExtraTreesClassifier,
                   'ADC'  : AdaBoostClassifier,
                   'SVC'  : SVC,
                   'SGDC' : SGDClassifier,
                   'GBC'  : GradientBoostingClassifier,
                   'MPC'  : MultilayerPerceptronClassifier,
                   'MNB'  : MultinomialNB
            }
    for item in selected_models:
        model_id, dataset = item.split(':')
        model = model_dict[model_id]()
        model.set_params(random_state = SEED)
        models.append((model, dataset))
    
    logger.info("Loading data...")
    _, y, _ = LoadData(); del _;
   
   for model, feature_set in models:
       model.set_pararms 
