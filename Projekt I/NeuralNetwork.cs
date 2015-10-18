namespace sieci_neuronowe
{
    #region

    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Text;

    using Encog;
    using Encog.Engine.Network.Activation;
    using Encog.ML;
    using Encog.ML.Data;
    using Encog.ML.Data.Basic;
    using Encog.ML.Data.Versatile;
    using Encog.ML.Data.Versatile.Columns;
    using Encog.ML.Data.Versatile.Sources;
    using Encog.ML.Factory;
    using Encog.ML.Model;
    using Encog.Neural.Networks;
    using Encog.Neural.Networks.Layers;
    using Encog.Neural.Networks.Training.Propagation.Back;
    using Encog.Util.CSV;

    #endregion

    public class NeuralNetwork
    {
        #region Constants

        private const string FirstColumn = "x";

        private const string GeneratedImagePath = @".\area_classification.bmp";
        
        private const string TrainingErrorDataPath = @".\training_error_data.txt";

        private const string VerificationErrorDataPath = @".\verification_error_data.txt";

        private const string PredictColumn = "cls";

        private const int RandomnessSeed = 1001;

        private const string SecondColumn = "y";

        #endregion

        #region Fields

        private readonly string learningPath;

        private readonly string logOutputPath;

        private readonly Random rng;

        private readonly string testingPath;

        #endregion

        #region Constructors and Destructors

        public NeuralNetwork(CommandLineParser parser)
            : this(parser.LearningSetFilePath, parser.TestingSetFilePath, parser.LogFilePath)
        {
        }

        public NeuralNetwork(string learningPath, string testingPath, string logOutputPath)
        {
            this.testingPath = testingPath ?? learningPath;
            this.learningPath = learningPath;
            this.logOutputPath = logOutputPath;
            this.rng = new Random(RandomnessSeed);
        }

        #endregion

        #region Public Methods and Operators

        public void Run()
        {
            // Loop over the entire, original, dataset and feed it through the model.
            // This also shows how you would process new data, that was not part of your
            // training set.  You do not need to retrain, simply use the NormalizationHelper
            // class.  After you train, you can save the NormalizationHelper to later
            // normalize and denormalize your data.
            var csvLearningDataSource = new CSVDataSource(this.learningPath, true, CSVFormat.DecimalPoint);
            VersatileMLDataSet dataSet = PrepareDataSet(csvLearningDataSource);
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
            var writetext = new StreamWriter(this.logOutputPath);

            trainingModel.Report = new StreamStatusReportable(writetext);

            // Now normalize the data.  Encog will automatically determine the correct normalization
            // type based on the model you chose in the last step.
            dataSet.Normalize();

            // Hold back some data for a final validation.
            // Shuffle the data into a random ordering.
            // Use a seed of 1001 so that we always use the same holdback and will get more consistent results.
            trainingModel.HoldBackValidation(0.3, true, RandomnessSeed);

            // Choose whatever is the default training type for this model.
            trainingModel.SelectTrainingType(dataSet);

            BasicNetwork network = CreateNetwork(this.rng);

            var trainingErrorWriter = new StreamWriter(TrainingErrorDataPath);
            var verificationErrorWriter = new StreamWriter(VerificationErrorDataPath);

            var backpropagation = new Backpropagation(network, dataSet, 0.00003, 0.001);
            backpropagation.BatchSize = 1; // Online
            const int IterationCount = 1000;
            for (int i = 0; i < IterationCount; i++)
            {
                backpropagation.Iteration();
                if (i % 100 == 0)
                {
                    trainingErrorWriter.WriteLine(CalcError(network, trainingModel.TrainingDataset));
                    verificationErrorWriter.WriteLine(CalcError(network, trainingModel.ValidationDataset));
                }
                if (i % (IterationCount / 10) != 0)
                {
                    continue;
                }

                double err = backpropagation.Error;
                writetext.WriteLine("Backpropagation error: " + err);
                Console.WriteLine("Iteration progress: " + i + "/" + IterationCount + ", error: " + err);
            }

            trainingErrorWriter.Close();
            verificationErrorWriter.Close();

            PrintWeights(network, writetext);
            var usedMethod = (BasicNetwork)backpropagation.Method;

            // Use a 5-fold cross-validated train.  Return the best method found.
            // var usedMethod = (IMLRegression)trainingModel.Crossvalidate(5, true);

            // Display our normalization parameters.
            NormalizationHelper normHelper = dataSet.NormHelper;
            writetext.WriteLine(normHelper);

            // Display the final model.
            writetext.WriteLine("Final model: " + usedMethod);
            /*
            writetext.WriteLine(
                "Training error: " + trainingModel.CalculateError(usedMethod, trainingModel.TrainingDataset));
            writetext.WriteLine(
                "Validation error: " + trainingModel.CalculateError(usedMethod, trainingModel.ValidationDataset));
            */

            writetext.WriteLine(
                "Training error: " + CalcError(usedMethod, trainingModel.TrainingDataset));
            writetext.WriteLine(
                "Validation error: " + CalcError(usedMethod, trainingModel.ValidationDataset));

            writetext.WriteLine("Neuron weight dump: " + network.DumpWeights());

            var allPoints = new List<NeuroPoint>();

            TestData(this.learningPath, normHelper, usedMethod, allPoints);
            TestData(this.testingPath, normHelper, usedMethod, allPoints);
            PrintPoints(allPoints, writetext);

            PictureGenerator.DrawArea(GeneratedImagePath, usedMethod, allPoints, normHelper, 1024, 1024);

            writetext.Close();

            EncogFramework.Instance.Shutdown();
        }

        #endregion

        private static double CalcError(BasicNetwork method, MatrixMLDataSet data)
        {
            int correct = 0;
            int total = 0;
            foreach (var pair in data)
            {
                var computed = method.Classify(pair.Input);
                for (int i = 0; i < pair.Ideal.Count; i++)
                {
                    if (computed == i && pair.Ideal[i] < 0.99999)
                    {
                        break;
                    }

                    if (pair.Ideal[i] > 0.99999)
                    {
                        break;
                    }

                    correct++;
                }

                total++;
            }

            return (total - correct) / (double)total;
        }

        #region Methods

        private static BasicNetwork CreateNetwork(Random rng)
        {
            var network = new BasicNetwork();

            // TODO: Wszystkie parametry konfigurowalne dla każdego layera (poza pierwszym bo input?)
            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 5));
            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 3));
            network.Structure.FinalizeStructure();

            // Zrób pełną sieć
            for (int i = 0; i < network.LayerCount - 1; i++)
            {
                for (int j = 0; j < network.GetLayerNeuronCount(i); j++)
                {
                    for (int k = 0; k < network.GetLayerNeuronCount(i + 1); k++)
                    {
                        network.SetWeight(i, j, k, (rng.NextDouble() - 0.5) * 2.0);
                    }
                }
            }

            return network;
        }

        private static VersatileMLDataSet PrepareDataSet(IVersatileDataSource dataSource)
        {
            var dataSet = new VersatileMLDataSet(dataSource);
            dataSet.DefineSourceColumn(FirstColumn, 0, ColumnType.Continuous);
            dataSet.DefineSourceColumn(SecondColumn, 1, ColumnType.Continuous);

            // Column that we are trying to predict.
            ColumnDefinition outputColumnDefinition = dataSet.DefineSourceColumn(PredictColumn, 2, ColumnType.Nominal);

            // Analyze the data, determine the min/max/mean/sd of every column.
            dataSet.Analyze();

            // Map the prediction column to the output of the model, and all other columns to the input.
            dataSet.DefineSingleOutputOthersInput(outputColumnDefinition);
            dataSet.LagWindowSize = 1;
            return dataSet;
        }

        private static void PrintPoints(IEnumerable<NeuroPoint> points, StreamWriter writetext)
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
            for (int i = 0; i < network.LayerCount - 1; i++)
            {
                writer.WriteLine("Layer {0}, neurons number {1}", i, network.GetLayerNeuronCount(i));
                for (int j = 0; j < network.GetLayerNeuronCount(i); j++)
                {
                    for (int k = 0; k < network.GetLayerNeuronCount(i + 1); k++)
                    {
                        writer.WriteLine("{0}->{1}: {2:F3}", j, k, network.GetWeight(i, j, k));
                    }
                }
            }
        }

        private static void TestData(
            string testedPath, 
            NormalizationHelper helper, 
            IMLRegression usedMethod, 
            ICollection<NeuroPoint> results)
        {
            var csv = new ReadCSV(testedPath, true, CSVFormat.DecimalPoint);

            while (csv.Next())
            {
                double x = csv.GetDouble(0);
                double y = csv.GetDouble(1);
                int correct = -1;
                if (csv.ColumnCount > 2)
                {
                    correct = (int)csv.GetDouble(2);
                }

                var data = new BasicMLData(new[] { x, y });
                helper.NormalizeInputVector(new[] { csv.Get(0), csv.Get(1) }, data.Data, false);
                IMLData output = usedMethod.Compute(data);
                string stringChosen = helper.DenormalizeOutputVectorToString(output)[0];
                int computed = int.Parse(stringChosen);
                results.Add(new NeuroPoint(x, y, computed, correct));
            }

            csv.Close();
        }

        #endregion
    }
}