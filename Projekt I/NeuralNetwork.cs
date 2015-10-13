using System;
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

namespace sieci_neuronowe
{
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

    public class NeuralNetwork
    {
        private string FirstColumn = "x";
        private string SecondColumn = "y";
        private string PredictColumn = "cls";

        public NeuralNetwork(string trainingPath, string testingPath, string outputPath)
        {
            IVersatileDataSource source = new CSVDataSource(trainingPath, false, CSVFormat.DecimalPoint);
            var data = new VersatileMLDataSet(source);
            data.DefineSourceColumn(FirstColumn, 0, ColumnType.Continuous);
            data.DefineSourceColumn(SecondColumn, 1, ColumnType.Continuous);

            // Column that we are trying to predict.

            var outputColumn = data.DefineSourceColumn(PredictColumn, 2, ColumnType.Nominal);

            // Analyze the data, determine the min/max/mean/sd of every column.

            data.Analyze();

            // Map the prediction column to the output of the model, and all other columns to the input.

            data.DefineSingleOutputOthersInput(outputColumn);

            // Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
            // You could also other model types, such as:
            // MLMethodFactory.SVM:  Support Vector Machine (SVM)
            // MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
            // MLMethodFactor.TYPE_NEAT: NEAT Neural Network
            // MLMethodFactor.TYPE_PNN: Probabilistic Neural Network

            var model = new EncogModel(data);
            model.SelectMethod(data, MLMethodFactory.TypeFeedforward);

            // Send any output to the console.
            var writetext = new StreamWriter(outputPath);
            model.Report = new StreamStatusReportable(writetext);

            // Now normalize the data.  Encog will automatically determine the correct normalization
            // type based on the model you chose in the last step.

            data.Normalize();

            // Hold back some data for a final validation.
            // Shuffle the data into a random ordering.
            // Use a seed of 1001 so that we always use the same holdback and will get more consistent results.

            model.HoldBackValidation(0.3, true, 1001);

            // Choose whatever is the default training type for this model.

            model.SelectTrainingType(data);

            // Use a 5-fold cross-validated train.  Return the best method found.

            var bestMethod = (IMLRegression)model.Crossvalidate(3, true);//true

            // Display the training and validation errors.

            writetext.WriteLine(@"Training error: " + model.CalculateError(bestMethod, model.TrainingDataset));
            writetext.WriteLine(@"Validation error: " + model.CalculateError(bestMethod, model.ValidationDataset));

            // Display our normalization parameters.

            var helper = data.NormHelper;
            writetext.WriteLine(helper.ToString());

            // Display the final model.

            writetext.WriteLine(@"Final model: " + bestMethod);

            // Loop over the entire, original, dataset and feed it through the model.
            // This also shows how you would process new data, that was not part of your
            // training set.  You do not need to retrain, simply use the NormalizationHelper
            // class.  After you train, you can save the NormalizationHelper to later
            // normalize and denormalize your data.

            source.Close();

            TestData(trainingPath, helper, bestMethod, writetext);
            TestData(testingPath, helper, bestMethod, writetext);

            writetext.WriteLine(@"Training error: " + model.CalculateError(bestMethod, model.TrainingDataset));
            writetext.WriteLine(@"Validation error: " + model.CalculateError(bestMethod, model.ValidationDataset));

            EncogFramework.Instance.Shutdown();

            writetext.Close();
        }

        private static void TestData(string testingPath, NormalizationHelper helper, IMLRegression bestMethod, TextWriter writetext)
        {
            var csv = new ReadCSV(testingPath, false, CSVFormat.DecimalPoint); //trainingPath
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

                result.AppendFormat("({0:F2}, {1:F2})", x, y);
                result.Append(" -> predicted: ");
                result.Append(irisChosen);
                if (correct != string.Empty)
                {
                    result.Append("(correct: ");
                    result.Append(correct);
                    result.Append(")");
                }

                writetext.WriteLine(result);
            }
            csv.Close();
        }
    }
}