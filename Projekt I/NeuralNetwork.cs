namespace sieci_neuronowe
{
    #region

    using System.IO;
    using System.Text;

    using Encog;
    using Encog.ML;
    using Encog.ML.Data;
    using Encog.ML.Data.Basic;
    using Encog.ML.Data.Versatile;
    using Encog.ML.Data.Versatile.Columns;
    using Encog.ML.Data.Versatile.Sources;
    using Encog.ML.Factory;
    using Encog.ML.Model;
    using Encog.Util.CSV;

    #endregion

    public class NeuralNetwork
    {
        #region Constants

        private const string FirstColumn = "x";

        private const string PredictColumn = "cls";

        private const string SecondColumn = "y";

        #endregion

        #region Fields

        private readonly string logOutputPath;

        private readonly string testingPath;

        private readonly string trainingPath;

        private IMLRegression bestMethod;

        private CSVDataSource csvTrainingDataSource;

        private VersatileMLDataSet dataSet;

        private NormalizationHelper normHelper;

        private ColumnDefinition outputColumnDefinition;

        private EncogModel trainingModel;

        #endregion

        #region Constructors and Destructors

        public NeuralNetwork(string trainingPath, string testingPath, string logOutputPath)
        {
            this.testingPath = testingPath ?? trainingPath;
            this.trainingPath = trainingPath;
            this.logOutputPath = logOutputPath;
        }

        #endregion

        #region Public Methods and Operators

        public void Run()
        {
            // TODO: dodać obsługę pierwszej linii z CSV (header = true w CSVDataSource).
            this.csvTrainingDataSource = new CSVDataSource(this.trainingPath, true, CSVFormat.DecimalPoint);
            this.dataSet = new VersatileMLDataSet(this.csvTrainingDataSource);
            this.dataSet.DefineSourceColumn(FirstColumn, 0, ColumnType.Continuous);
            this.dataSet.DefineSourceColumn(SecondColumn, 1, ColumnType.Continuous);

            // Column that we are trying to predict.
            this.outputColumnDefinition = this.dataSet.DefineSourceColumn(PredictColumn, 2, ColumnType.Nominal);

            // Analyze the data, determine the min/max/mean/sd of every column.
            this.dataSet.Analyze();

            // Map the prediction column to the output of the model, and all other columns to the input.
            this.dataSet.DefineSingleOutputOthersInput(this.outputColumnDefinition);

            // Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
            // You could also other model types, such as:
            // MLMethodFactory.SVM:  Support Vector Machine (SVM)
            // MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
            // MLMethodFactor.TYPE_NEAT: NEAT Neural Network
            // MLMethodFactor.TYPE_PNN: Probabilistic Neural Network
            this.trainingModel = new EncogModel(this.dataSet);
            this.trainingModel.SelectMethod(this.dataSet, MLMethodFactory.TypeFeedforward);

            // Send any output to the console.
            using (var writetext = new StreamWriter(this.logOutputPath))
            {
                this.trainingModel.Report = new StreamStatusReportable(writetext);

                // Now normalize the data.  Encog will automatically determine the correct normalization
                // type based on the model you chose in the last step.
                this.dataSet.Normalize();

                // Hold back some data for a final validation.
                // Shuffle the data into a random ordering.
                // Use a seed of 1001 so that we always use the same holdback and will get more consistent results.
                this.trainingModel.HoldBackValidation(0.3, true, 1001);

                // Choose whatever is the default training type for this model.
                this.trainingModel.SelectTrainingType(this.dataSet);

                // Use a 5-fold cross-validated train.  Return the best method found.
                this.bestMethod = (IMLRegression)this.trainingModel.Crossvalidate(10, true);

                // Display the training and validation errors.
                writetext.WriteLine(
                    @"Training error: "
                    + this.trainingModel.CalculateError(this.bestMethod, this.trainingModel.TrainingDataset));
                writetext.WriteLine(
                    @"Validation error: "
                    + this.trainingModel.CalculateError(this.bestMethod, this.trainingModel.ValidationDataset));

                // Display our normalization parameters.
                this.normHelper = this.dataSet.NormHelper;
                writetext.WriteLine(this.normHelper);

                // Display the final model.
                writetext.WriteLine(@"Final model: " + this.bestMethod);

                // Loop over the entire, original, dataset and feed it through the model.
                // This also shows how you would process new data, that was not part of your
                // training set.  You do not need to retrain, simply use the NormalizationHelper
                // class.  After you train, you can save the NormalizationHelper to later
                // normalize and denormalize your data.
                this.csvTrainingDataSource.Close();

                this.TestData(this.normHelper, writetext);
                this.TestData(this.normHelper, writetext);

                writetext.WriteLine(
                    @"Training error: "
                    + this.trainingModel.CalculateError(this.bestMethod, this.trainingModel.TrainingDataset));
                writetext.WriteLine(
                    @"Validation error: "
                    + this.trainingModel.CalculateError(this.bestMethod, this.trainingModel.ValidationDataset));

                PictureGenerator.DrawArea(
                    "area_classification.bmp", 
                    this.bestMethod, 
                    this.trainingModel.TrainingDataset, 
                    this.normHelper, 
                    1024, 
                    1024);

                EncogFramework.Instance.Shutdown();
            }
        }

        #endregion

        #region Methods

        private void TestData(NormalizationHelper helper, TextWriter writetext)
        {
            var csv = new ReadCSV(this.testingPath, true, CSVFormat.DecimalPoint);
            IMLData input = helper.AllocateInputVector();

            while (csv.Next())
            {
                var result = new StringBuilder();
                double x = csv.GetDouble(0);
                double y = csv.GetDouble(1);
                string correct = string.Empty;
                if (csv.ColumnCount > 2)
                {
                    correct = csv.Get(2);
                }

                helper.NormalizeInputVector(new[] { csv.Get(0), csv.Get(1) }, ((BasicMLData)input).Data, false);
                IMLData output = this.bestMethod.Compute(input);
                string irisChosen = helper.DenormalizeOutputVectorToString(output)[0];

                // "Dziwny" format żeby długość linii była taka sama (i "predicted" było w tym samym miejscu)
                result.AppendFormat("({0: 0.00;-0.00}, {1: 0.00;-0.00}) -> predicted: {2}", x, y, irisChosen);
                if (correct != string.Empty)
                {
                    result.AppendFormat("(correct: {0})", correct);
                }

                writetext.WriteLine(result);
            }

            csv.Close();
        }

        #endregion
    }
}