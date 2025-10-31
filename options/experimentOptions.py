from .baseOptions import BaseOptions
from .GANOptions import BaseGANOptions

class TrainOptions(BaseOptions):
    def initialize(self, json_filepath):
        BaseOptions.initialize(self, json_filepath)
        # any options to adjust in training (none for now)
        self.parser.add_argument('--output_csv', type=bool, default=True, help='generate a final csv of the predictions on train and val sets?')

        self.isTrain = True

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # any options to adjust in training (none for now)
        self.parser.add_argument('--output_train_val', type=bool, default=False, help='run eval on train and val sets as well?')

        self.isTrain = False

class GANTrainOptions(BaseGANOptions):
    def initialize(self):
        BaseGANOptions.initialize(self)

        self.isTrain = True