using Encog;
using Encog.ML;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Versatile;
using Encog.ML.Data.Versatile.Columns;
using Encog.ML.Data.Versatile.Sources;
using Encog.ML.Factory;
using Encog.ML.Model;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Util.CSV;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace sieci_neuronowe
{
    public class NeuralNetwork
    {
        private const string FirstColumn = "x";
        private const string SecondColumn = "y";
        private const string PredictColumn = "cls";
        private const string GeneratedImagePath = @".\area_classification.bmp";
        private const string TrainingErrorDataPath = @".\training_error_data.txt";
        private const string VerificationErrorDataPath = @".\verification_error_data.txt";
        private const int RandomnessSeed = 1001;
        private const double ValidationPercent = 0.3;
        private const double LearnRate = 0.0003;
        private const int BackpropagationBatchSize = 1;
        private const int ImageResolutionX = 1024;
        private const int ImageResolutionY = 1024;
        private const bool IfShuffle = true;
        private readonly string learningPath;
        private readonly string logOutputPath;
        private readonly BasicNetwork neuralNetwork;
        private readonly string testingPath;
        private readonly ArgsParser parser;

        public NeuralNetwork(ArgsParser inParser, BasicNetwork neuralNetwork)
        {
            this.parser = inParser;

            this.testingPath = parser.TestingSetFilePath ?? parser.LearningSetFilePath;
            this.learningPath = parser.LearningSetFilePath;
            this.logOutputPath = parser.LogFilePath;
            this.neuralNetwork = neuralNetwork;
        }

        public void Run()
        {
            // Loop over the entire, original, dataset and feed it through the model.
            // This also shows how you would process new data, that was not part of your
            // training set.  You do not need to retrain, simply use the NormalizationHelper
            // class.  After you train, you can save the NormalizationHelper to later
            // normalize and denormalize your data.

            var csvLearningDataSource = new CSVDataSource(learningPath, true, CSVFormat.DecimalPoint);
            var problemType = parser.Problem;
            var dataSet = problemType == ArgsParser.ProblemType.Regression
                          ? PrepareRegressionDataSet(csvLearningDataSource)
                          : PrepareClassificationDataSet(csvLearningDataSource);
            csvLearningDataSource.Close();

            // Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
            // You could also other model types, such as:
            // MLMethodFactory.SVM:  Support Vector Machine (SVM)
            // MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
            // MLMethodFactor.TYPE_NEAT: NEAT Neural Network
            // MLMethodFactor.TYPE_PNN: Probabilistic Neural Network

            var trainingModel = new EncogModel(dataSet);
            trainingModel.SelectMethod(dataSet, MLMethodFactory.TypeFeedforward);

            // Send any output to the console.

            var writetext = new StreamWriter(logOutputPath);
            trainingModel.Report = new StreamStatusReportable(writetext);

            // Now normalize the data.  Encog will automatically determine the correct normalization
            // type based on the model you chose in the last step.

            dataSet.Normalize();

            // Hold back some data for a final validation.
            // Shuffle the data into a random ordering.

            trainingModel.HoldBackValidation(ValidationPercent, IfShuffle, RandomnessSeed);

            // Choose whatever is the default training type for this model.

            trainingModel.SelectTrainingType(dataSet);

            var trainingErrorWriter = new StreamWriter(TrainingErrorDataPath);
            var verificationErrorWriter = new StreamWriter(VerificationErrorDataPath);
            var backpropagation = new Backpropagation(this.neuralNetwork, dataSet, LearnRate, this.parser.Momentum) { BatchSize = BackpropagationBatchSize };

            var iterationsNumber = parser.NumberOfIterations;
            for (var i = 0; i < iterationsNumber; i++)
            {
                backpropagation.Iteration();
                if (i % 100 == 0)
                {
                    trainingErrorWriter.WriteLine(this.CalcError(this.neuralNetwork, trainingModel.TrainingDataset));
                    verificationErrorWriter.WriteLine(this.CalcError(this.neuralNetwork, trainingModel.ValidationDataset));
                }

                if (i % (iterationsNumber / 10) != 0)
                {
                    continue;
                }

                var err = backpropagation.Error;
                writetext.WriteLine("Backpropagation error: " + err);
                Console.WriteLine("Iteration progress: {0} / {1}, error = {2}", i, iterationsNumber, err);
            }

            trainingErrorWriter.Close();
            verificationErrorWriter.Close();

            PrintWeights(this.neuralNetwork, writetext);
            var usedMethod = (BasicNetwork)backpropagation.Method;

            // Use a 5-fold cross-validated train.  Return the best method found.
            // var usedMethod = (BasicNetwork)trainingModel.Crossvalidate(5, true);

            // Display our normalization parameters.
            var normHelper = dataSet.NormHelper;
            writetext.WriteLine(normHelper);

            // Display the final model.
            writetext.WriteLine("Final model: " + usedMethod);

            writetext.WriteLine("Training error: " + CalcError(usedMethod, trainingModel.TrainingDataset));
            writetext.WriteLine("Validation error: " + CalcError(usedMethod, trainingModel.ValidationDataset));
            writetext.WriteLine("Neuron weight dump: " + this.neuralNetwork.DumpWeights());

            var allPoints = new List<ClassifiedPoint>();
            TestData(learningPath, normHelper, usedMethod, allPoints);
            TestData(testingPath, normHelper, usedMethod, allPoints);
            PrintPoints(allPoints, writetext);
            DrawPicture(GeneratedImagePath, usedMethod, allPoints, normHelper, ImageResolutionX, ImageResolutionY);

            writetext.Close();

            EncogFramework.Instance.Shutdown();
        }

        public static int ActualCategory(IMLData pt)
        {
            // Returns index of the value at which pt is almost 1 if there is exactly one such value
            // Otherwise returns -1
            const double GoodEnough = 0.9;
            int category = -1;
            for (var i = 0; i < pt.Count; i++)
            {
                if (pt[i] > GoodEnough)
                {
                    if (category >= 0)
                    {
                        return -1;
                    }
                    category = i;
                    continue;
                }

                if (pt[i] > -GoodEnough)
                {
                    return -1;
                }
            }

            return category;
        }

        private static double ClassificationError(BasicNetwork method, IEnumerable<IMLDataPair> data)
        {
            var correct = 0;
            var total = 0;
            foreach (var pair in data)
            {
                var computed = method.Classify(pair.Input);
                var actual = ActualCategory(pair.Ideal);
                if (computed == actual)
                {
                    correct++;
                }

                total++;
            }

            return (total - correct) / (double)total;
        }

        private static VersatileMLDataSet PrepareClassificationDataSet(IVersatileDataSource dataSource)
        {
            var dataSet = new VersatileMLDataSet(dataSource);
            dataSet.DefineSourceColumn(FirstColumn, 0, ColumnType.Continuous);
            dataSet.DefineSourceColumn(SecondColumn, 1, ColumnType.Continuous);

            // Column that we are trying to predict.
            var outputColumnDefinition = dataSet.DefineSourceColumn(PredictColumn, 2, ColumnType.Nominal);

            // Analyze the data, determine the min/max/mean/sd of every column.
            dataSet.Analyze();

            // Map the prediction column to the output of the model, and all other columns to the input.
            dataSet.DefineSingleOutputOthersInput(outputColumnDefinition);
            dataSet.LagWindowSize = 1;
            return dataSet;
        }

        private static VersatileMLDataSet PrepareRegressionDataSet(IVersatileDataSource dataSource)
        {
            var dataSet = new VersatileMLDataSet(dataSource);
            dataSet.DefineSourceColumn(FirstColumn, 0, ColumnType.Continuous);

            // Column that we are trying to predict.
            ColumnDefinition outputColumnDefinition = dataSet.DefineSourceColumn(SecondColumn, 1, ColumnType.Continuous);

            // Analyze the data, determine the min/max/mean/sd of every column.
            dataSet.Analyze();

            // Map the prediction column to the output of the model, and all other columns to the input.
            dataSet.DefineSingleOutputOthersInput(outputColumnDefinition);
            dataSet.LagWindowSize = 1;
            return dataSet;
        }

        private static void PrintPoints(IEnumerable<ClassifiedPoint> points, StreamWriter writetext)
        {
            foreach (var point in points)
            {
                var result = new StringBuilder();

                result.AppendFormat(
                    "({0: 0.00;-0.00}, {1: 0.00;-0.00}) -> predicted: {2}",
                    point.X,
                    point.Y,
                    point.Category);
                if (point.Correct >= 0)
                {
                    result.AppendFormat("(correct: {0})", point.Correct);
                }

                writetext.WriteLine(result);
            }
        }

        private static void PrintWeights(BasicNetwork network, TextWriter writer)
        {
            for (var i = 0; i < network.LayerCount - 1; i++)
            {
                writer.WriteLine("Layer: {0}. Neurons number: {1}.", i, network.GetLayerNeuronCount(i));
                for (var j = 0; j < network.GetLayerNeuronCount(i); j++)
                {
                    for (var k = 0; k < network.GetLayerNeuronCount(i + 1); k++)
                    {
                        writer.WriteLine("{0}->{1}: {2:F3}", j, k, network.GetWeight(i, j, k));
                    }
                }
            }
        }

        private static double RegressionError(BasicNetwork method, IEnumerable<IMLDataPair> data)
        {
            var error = 0.0;
            var total = 0;
            foreach (var pair in data)
            {
                var computed = method.Compute(pair.Input);
                var min = Math.Min(computed.Count, pair.Ideal.Count);
                var max = Math.Max(computed.Count, pair.Ideal.Count);
                for (var i = 0; i < min; i++)
                {
                    error += Math.Abs(computed[i] - pair.Ideal[i]);
                }
                error += max - min;
                total += max;
            }

            return error / total;
        }

        private static void TestClassificationData(
            string testedPath,
            NormalizationHelper helper,
            BasicNetwork usedMethod,
            ICollection<ClassifiedPoint> results)
        {
            var csv = new ReadCSV(testedPath, true, CSVFormat.DecimalPoint);

            while (csv.Next())
            {
                var x = csv.GetDouble(0);
                var y = csv.GetDouble(1);
                var correct = -1;
                if (csv.ColumnCount > 2)
                {
                    correct = (int)csv.GetDouble(2);
                }

                var data = new BasicMLData(new[] { x, y });
                helper.NormalizeInputVector(new[] { csv.Get(0), csv.Get(1) }, data.Data, false);
                var output = usedMethod.Compute(data);
                var stringChosen = helper.DenormalizeOutputVectorToString(output)[0];
                var computed = int.Parse(stringChosen);
                results.Add(new ClassifiedPoint(x, y, computed, correct));
            }

            csv.Close();
        }

        public static double[] DataToArray(IMLData data)
        {
            double[] ret = new double[data.Count];
            data.CopyTo(ret, 0, data.Count);
            return ret;
        }

        private static void TestRegressionData(
            string testedPath,
            NormalizationHelper helper,
            BasicNetwork usedMethod,
            ICollection<ClassifiedPoint> results)
        {
            var csv = new ReadCSV(testedPath, true, CSVFormat.DecimalPoint);

            while (csv.Next())
            {
                var x = csv.GetDouble(0);

                var data = new BasicMLData(new[] { x });
                helper.NormalizeInputVector(new[] { csv.Get(0) }, data.Data, false);
                var output = usedMethod.Compute(data);
                var outstr = helper.DenormalizeOutputVectorToString(output);
                var y = double.Parse(outstr[0]);
                results.Add(new ClassifiedPoint(x, y, -1, -1));
            }

            csv.Close();
        }

        private double CalcError(BasicNetwork method, IEnumerable<IMLDataPair> data)
        {
            return parser.Problem == ArgsParser.ProblemType.Regression
                       ? RegressionError(method, data)
                       : ClassificationError(method, data);
        }

        private void DrawPicture(
            string path,
            IMLRegression testFunction,
            List<ClassifiedPoint> points,
            NormalizationHelper helper,
            int resolutionX,
            int resolutionY)
        {
            if (parser.Problem == ArgsParser.ProblemType.Regression)
            {
                PictureGenerator.DrawGraph(path, testFunction, points, helper, resolutionX, resolutionY);
            }
            else
            {
                PictureGenerator.DrawArea(path, testFunction, points, helper, resolutionX, resolutionY);
            }
        }

        private void TestData(
            string testedPath,
            NormalizationHelper helper,
            BasicNetwork usedMethod,
            ICollection<ClassifiedPoint> results)
        {
            if (parser.Problem == ArgsParser.ProblemType.Regression)
            {
                TestRegressionData(testedPath, helper, usedMethod, results);
            }
            else
            {
                TestClassificationData(testedPath, helper, usedMethod, results);
            }
        }
    }
}