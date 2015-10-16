namespace sieci_neuronowe
{
    #region

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
    using Encog.Util.CSV;

    #endregion

    public struct NeuroPoint
    {
        #region Fields

        public int Category;

        public int Correct;

        public double X;

        public double Y;

        #endregion

        #region Constructors and Destructors

        public NeuroPoint(double x, double y, int category, int correct)
        {
            this.X = x;
            this.Y = y;
            this.Category = category;
            this.Correct = correct;
        }

        #endregion
    }

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
            // Loop over the entire, original, dataset and feed it through the model.
            // This also shows how you would process new data, that was not part of your
            // training set.  You do not need to retrain, simply use the NormalizationHelper
            // class.  After you train, you can save the NormalizationHelper to later
            // normalize and denormalize your data.
            var csvTrainingDataSource = new CSVDataSource(this.trainingPath, true, CSVFormat.DecimalPoint);
            VersatileMLDataSet dataSet = PrepareDataSet(csvTrainingDataSource);
            csvTrainingDataSource.Close();

            // Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
            // You could also other model types, such as:
            // MLMethodFactory.SVM:  Support Vector Machine (SVM)
            // MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
            // MLMethodFactor.TYPE_NEAT: NEAT Neural Network
            // MLMethodFactor.TYPE_PNN: Probabilistic Neural Network
            var trainingModel = new EncogModel(dataSet);
            trainingModel.SelectMethod(dataSet, MLMethodFactory.TypeFeedforward);

            // Send any output to the console.
            using (var writetext = new StreamWriter(this.logOutputPath))
            {
                trainingModel.Report = new StreamStatusReportable(writetext);

                // Now normalize the data.  Encog will automatically determine the correct normalization
                // type based on the model you chose in the last step.
                dataSet.Normalize();

                // Hold back some data for a final validation.
                // Shuffle the data into a random ordering.
                // Use a seed of 1001 so that we always use the same holdback and will get more consistent results.
                trainingModel.HoldBackValidation(0.3, true, 1001);

                // Choose whatever is the default training type for this model.
                trainingModel.SelectTrainingType(dataSet);

                /*
                BasicNetwork network = CreateNetwork();

                // TODO: Ma być online, tzn. training dataset pusty (niemożliwe z tą implementacją?)
                var backpropagation = new Backpropagation(network, dataSet, 0.000001, 0.0001);
                for (int i = 0; i < 100; i++)
                {
                    writetext.WriteLine(@"Backpropagation error: " + backpropagation.Error);
                    backpropagation.Iteration(100);
                }
                 */

                // var usedMethod = (IMLRegression)backpropagation.Method;

                // Use a 5-fold cross-validated train.  Return the best method found.
                var usedMethod = (IMLRegression)trainingModel.Crossvalidate(5, true);

                // Display the training and validation errors.
                writetext.WriteLine(
                    @"Training error: " + trainingModel.CalculateError(usedMethod, trainingModel.TrainingDataset));
                writetext.WriteLine(
                    @"Validation error: " + trainingModel.CalculateError(usedMethod, trainingModel.ValidationDataset));

                // Display our normalization parameters.
                NormalizationHelper normHelper = dataSet.NormHelper;
                writetext.WriteLine(normHelper);

                // Display the final model.
                writetext.WriteLine(@"Final model: " + usedMethod);

                var allPoints = new List<NeuroPoint>();

                TestData(this.trainingPath, normHelper, usedMethod, allPoints);
                TestData(this.testingPath, normHelper, usedMethod, allPoints);
                PrintPoints(allPoints, writetext);

                writetext.WriteLine(
                    @"Training error: " + trainingModel.CalculateError(usedMethod, trainingModel.TrainingDataset));
                writetext.WriteLine(
                    @"Validation error: " + trainingModel.CalculateError(usedMethod, trainingModel.ValidationDataset));

                PictureGenerator.DrawArea(
                    "area_classification.bmp", 
                    usedMethod, 
                    allPoints, 
                    normHelper, 
                    1024, 
                    1024);

                // writetext.WriteLine(network.DumpWeights());
            }

            EncogFramework.Instance.Shutdown();
        }

        #endregion

        #region Methods

        private static BasicNetwork CreateNetwork()
        {
            var network = new BasicNetwork();

            // TODO: Wszystkie parametry konfigurowalne dla każdego layera (poza pierwszym bo input?)
            var layer = new BasicLayer(new ActivationLinear(), false, 2);
            network.AddLayer(layer);
            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 8));
            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 1));
            network.Structure.FinalizeStructure();

            // Zrób pełną sieć
            for (int i = 0; i < network.LayerCount - 1; i++)
            {
                for (int j = 0; j < network.GetLayerNeuronCount(i); j++)
                {
                    for (int k = 0; k < network.GetLayerNeuronCount(i + 1); k++)
                    {
                        network.SetWeight(i, j, k, 1.0);
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

                // "Dziwny" format żeby długość linii była taka sama (i "predicted" było w tym samym miejscu)
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

        private static void TestData(
            string testedPath, 
            NormalizationHelper helper, 
            IMLRegression usedMethod, 
            List<NeuroPoint> results)
        {
            var csv = new ReadCSV(testedPath, true, CSVFormat.DecimalPoint);
            IMLData input = helper.AllocateInputVector();

            while (csv.Next())
            {
                double x = csv.GetDouble(0);
                double y = csv.GetDouble(1);
                int correct = -1;
                if (csv.ColumnCount > 2)
                {
                    correct = (int)csv.GetDouble(2);
                }

                helper.NormalizeInputVector(new[] { csv.Get(0), csv.Get(1) }, ((BasicMLData)input).Data, false);
                IMLData output = usedMethod.Compute(input);
                string stringChosen = helper.DenormalizeOutputVectorToString(output)[0];
                int computed = int.Parse(stringChosen);
                results.Add(new NeuroPoint(x, y, computed, correct));
            }

            csv.Close();
        }

        #endregion
    }
}