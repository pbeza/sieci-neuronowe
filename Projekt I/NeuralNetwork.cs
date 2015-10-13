using System;
using System.IO;
using System.Text;

using Encog;
using Encog.ML;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Versatile;
using Encog.ML.Data.Versatile.Columns;
using Encog.ML.Data.Versatile.Sources;
using Encog.ML.Factory;
using Encog.ML.Model;
using Encog.Util.CSV;

namespace sieci_neuronowe
{
    public class NeuralNetwork
    {
        protected const string FirstColumn = "x";
        protected const string SecondColumn = "y";
        protected const string PredictColumn = "cls";

        protected string TestingPath;
        protected string TrainingPath;
        protected string LogOutputPath;
        protected CSVDataSource CsvTrainingDataSource;
        protected VersatileMLDataSet DataSet;
        protected ColumnDefinition OutputColumnDefinition;
        protected EncogModel TrainingModel;
        protected IMLRegression BestMethod;
        protected NormalizationHelper NormHelper;

        public NeuralNetwork(string trainingPath, string testingPath, string logOutputPath)
        {
            TestingPath = testingPath ?? trainingPath;
            TrainingPath = trainingPath;
            LogOutputPath = logOutputPath;
        }

        public void Run()
        {
            // TODO: dodać obsługę pierwszej linii z CSV (header = true w CSVDataSource).
            CsvTrainingDataSource = new CSVDataSource(TrainingPath, true, CSVFormat.DecimalPoint);
            DataSet = new VersatileMLDataSet(CsvTrainingDataSource);
            DataSet.DefineSourceColumn(FirstColumn, 0, ColumnType.Continuous);
            DataSet.DefineSourceColumn(SecondColumn, 1, ColumnType.Continuous);

            // Column that we are trying to predict.

            OutputColumnDefinition = DataSet.DefineSourceColumn(PredictColumn, 2, ColumnType.Nominal);

            // Analyze the data, determine the min/max/mean/sd of every column.

            DataSet.Analyze();

            // Map the prediction column to the output of the model, and all other columns to the input.

            DataSet.DefineSingleOutputOthersInput(OutputColumnDefinition);

            // Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
            // You could also other model types, such as:
            // MLMethodFactory.SVM:  Support Vector Machine (SVM)
            // MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
            // MLMethodFactor.TYPE_NEAT: NEAT Neural Network
            // MLMethodFactor.TYPE_PNN: Probabilistic Neural Network

            TrainingModel = new EncogModel(DataSet);
            TrainingModel.SelectMethod(DataSet, MLMethodFactory.TypeFeedforward);

            // Send any output to the console.
            using (var writetext = new StreamWriter(LogOutputPath))
            {
                TrainingModel.Report = new StreamStatusReportable(writetext);

                // Now normalize the data.  Encog will automatically determine the correct normalization
                // type based on the model you chose in the last step.

                DataSet.Normalize();

                // Hold back some data for a final validation.
                // Shuffle the data into a random ordering.
                // Use a seed of 1001 so that we always use the same holdback and will get more consistent results.

                TrainingModel.HoldBackValidation(0.3, true, 1001);

                // Choose whatever is the default training type for this model.

                TrainingModel.SelectTrainingType(DataSet);

                // Use a 5-fold cross-validated train.  Return the best method found.

                BestMethod = (IMLRegression)TrainingModel.Crossvalidate(3, true);

                // Display the training and validation errors.

                writetext.WriteLine(@"Training error: " + TrainingModel.CalculateError(BestMethod, TrainingModel.TrainingDataset));
                writetext.WriteLine(@"Validation error: " + TrainingModel.CalculateError(BestMethod, TrainingModel.ValidationDataset));

                // Display our normalization parameters.

                NormHelper = DataSet.NormHelper;
                writetext.WriteLine(NormHelper);

                // Display the final model.

                writetext.WriteLine(@"Final model: " + BestMethod);

                // Loop over the entire, original, dataset and feed it through the model.
                // This also shows how you would process new data, that was not part of your
                // training set.  You do not need to retrain, simply use the NormalizationHelper
                // class.  After you train, you can save the NormalizationHelper to later
                // normalize and denormalize your data.

                CsvTrainingDataSource.Close();

                TestData(NormHelper, BestMethod, writetext);
                TestData(NormHelper, BestMethod, writetext);

                writetext.WriteLine(@"Training error: " + TrainingModel.CalculateError(BestMethod, TrainingModel.TrainingDataset));
                writetext.WriteLine(@"Validation error: " + TrainingModel.CalculateError(BestMethod, TrainingModel.ValidationDataset));

                EncogFramework.Instance.Shutdown();
            }
        }

        private void TestData(NormalizationHelper helper, IMLRegression bestMethod, TextWriter writetext)
        {
            var csv = new ReadCSV(TestingPath, true, CSVFormat.DecimalPoint);
            var input = helper.AllocateInputVector();

            while (csv.Next())
            {
                var result = new StringBuilder();
                var x = csv.GetDouble(0);
                var y = csv.GetDouble(1);
                var correct = string.Empty;
                if (csv.ColumnCount > 2)
                    correct = csv.Get(2);
                helper.NormalizeInputVector(new[] { csv.Get(0), csv.Get(1) }, ((BasicMLData)input).Data, false);
                var output = bestMethod.Compute(input);
                var irisChosen = helper.DenormalizeOutputVectorToString(output)[0];

                result.AppendFormat("({0:F2}, {1:F2}) -> predicted: {2}", x, y, irisChosen);
                if (correct != string.Empty)
                {
                    result.AppendFormat("(correct: {0})", correct);
                }

                writetext.WriteLine(result);
            }
            csv.Close();
        }

        public class StreamStatusReportable : IStatusReportable
        {
            private readonly StreamWriter _writter;

            public StreamStatusReportable(StreamWriter writter)
            {
                _writter = writter;
            }

            public void Report(int total, int current,
                               String message)
            {
                if (total == 0)
                {
                    _writter.WriteLine(current + " : " + message);
                }
                else
                {
                    _writter.WriteLine(current + "/" + total + " : " + message);
                }
            }
        }
    }
}