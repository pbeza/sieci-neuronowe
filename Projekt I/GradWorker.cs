namespace sieci_neuronowe
{
    #region

    using System;

    using Encog.Engine.Network.Activation;
    using Encog.MathUtil.Error;
    using Encog.ML.Data;
    using Encog.Neural.Error;
    using Encog.Neural.Flat;
    using Encog.Util;

    #endregion

    public class GradWorker
    {
        private readonly double[] _actual;

        private readonly ErrorCalculation _errorCalculation;

        private readonly double[] _gradients;

        private readonly int _high;

        private readonly int[] _layerCounts;

        private readonly double[] _layerDelta;

        private readonly int[] _layerFeedCounts;

        private readonly int[] _layerIndex;

        private readonly double[] _layerOutput;

        private readonly double[] _layerSums;

        private readonly int _low;

        private readonly FlatNetwork _network;

        private readonly BackProp _owner;

        private readonly IMLDataSet _training;

        private readonly int[] _weightIndex;

        private readonly double[] _weights;

        private readonly double[] _flatSpot;

        private readonly IErrorFunction _ef;

        public GradWorker(
            FlatNetwork theNetwork,
            BackProp theOwner,
            IMLDataSet theTraining,
            int theLow,
            int theHigh,
            double[] theFlatSpots,
            IErrorFunction ef)
        {
            _errorCalculation = new ErrorCalculation();
            _network = theNetwork;
            _training = theTraining;
            _low = theLow;
            _high = theHigh;
            _owner = theOwner;
            _flatSpot = theFlatSpots;

            _layerDelta = new double[_network.LayerOutput.Length];
            _gradients = new double[_network.Weights.Length];
            _actual = new double[_network.OutputCount];

            _weights = _network.Weights;
            _layerIndex = _network.LayerIndex;
            _layerCounts = _network.LayerCounts;
            _weightIndex = _network.WeightIndex;
            _layerOutput = _network.LayerOutput;
            _layerSums = _network.LayerSums;
            _layerFeedCounts = _network.LayerFeedCounts;
            _ef = ef;
        }

        #region FlatGradientWorker Members

        public FlatNetwork Network
        {
            get
            {
                return _network;
            }
        }

        public double[] Weights
        {
            get
            {
                return _weights;
            }
        }

        public void Run()
        {
            try
            {
                _errorCalculation.Reset();
                for (int i = _low; i <= _high; i++)
                {
                    var pair = _training[i];
                    Process(pair);
                }
                double error = _errorCalculation.Calculate();
                _owner.Report(_gradients, error, null);
                EngineArray.Fill(_gradients, 0);
            }
            catch (Exception ex)
            {
                _owner.Report(null, 0, ex);
            }
        }

        #endregion

        private void Process(IMLDataPair pair)
        {
            _network.Compute(pair.Input, _actual);

            _errorCalculation.UpdateError(_actual, pair.Ideal, pair.Significance);
            _ef.CalculateError(pair.Ideal, _actual, _layerDelta);

            for (int i = 0; i < _actual.Length; i++)
            {
                _layerDelta[i] = (_network.ActivationFunctions[0].DerivativeFunction(_layerSums[i], _layerOutput[i])
                                  + _flatSpot[0]) * _layerDelta[i] * pair.Significance;
            }

            for (int i = _network.BeginTraining; i < _network.EndTraining; i++)
            {
                ProcessLevel(i);
            }
        }

        public ErrorCalculation CalculateError
        {
            get
            {
                return _errorCalculation;
            }
        }

        public void Run(int index)
        {
            IMLDataPair pair = _training[index];
            Process(pair);
            _owner.Report(_gradients, 0, null);
            EngineArray.Fill(_gradients, 0);
        }

        private void ProcessLevel(int currentLevel)
        {
            int fromLayerIndex = _layerIndex[currentLevel + 1];
            int toLayerIndex = _layerIndex[currentLevel];
            int fromLayerSize = _layerCounts[currentLevel + 1];
            int toLayerSize = _layerFeedCounts[currentLevel];

            int index = _weightIndex[currentLevel];
            IActivationFunction activation = _network.ActivationFunctions[currentLevel + 1];
            double currentFlatSpot = _flatSpot[currentLevel + 1];

            // handle weights
            int yi = fromLayerIndex;
            for (int y = 0; y < fromLayerSize; y++)
            {
                double output = _layerOutput[yi];
                double sum = 0;
                int xi = toLayerIndex;
                int wi = index + y;
                for (int x = 0; x < toLayerSize; x++)
                {
                    _gradients[wi] += output * _layerDelta[xi];
                    sum += _weights[wi] * _layerDelta[xi];
                    wi += fromLayerSize;
                    xi++;
                }

                _layerDelta[yi] = sum
                                  * (activation.DerivativeFunction(_layerSums[yi], _layerOutput[yi]) + currentFlatSpot);
                yi++;
            }
        }
    }
}