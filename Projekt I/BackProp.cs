namespace sieci_neuronowe
{
    #region

    using System;
    using System.Threading.Tasks;

    using Encog;
    using Encog.Engine.Network.Activation;
    using Encog.MathUtil;
    using Encog.ML;
    using Encog.ML.Data;
    using Encog.ML.Train;
    using Encog.Neural.Error;
    using Encog.Neural.Flat;
    using Encog.Neural.Networks;
    using Encog.Neural.Networks.Training;
    using Encog.Neural.Networks.Training.Propagation;
    using Encog.Util;
    using Encog.Util.Concurrency;
    using Encog.Util.Logging;
    using Encog.Util.Validate;

    #endregion

    public sealed class BackProp : BasicTraining, ITrain, IMultiThreadable, IBatchSize
    {
        private readonly Random rng;
        private readonly IMLDataSet _indexable;

        private readonly double[] _lastGradient;

        private readonly IContainsFlat _network;

        private readonly FlatNetwork _flat;

        private readonly IMLDataSet _training;

        internal double[] Gradients;

        private int _iteration;

        private int _numThreads;

        private Exception _reportedException;

        private double _totalError;

        private GradWorker[] _workers;

        public bool FixFlatSpot { get; set; }

        private double[] _flatSpot;

        public IErrorFunction ErrorFunction { get; set; }


        public int ThreadCount
        {
            get
            {
                return _numThreads;
            }
            set
            {
                _numThreads = value;
            }
        }

        public void RollIteration()
        {
            _iteration++;
        }

        #region Train Members

        public override IMLMethod Method
        {
            get
            {
                return _network;
            }
        }

        public override void Iteration()
        {
            try
            {
                PreIteration();

                RollIteration();

                if (BatchSize == 0)
                {
                    ProcessPureBatch();
                }
                else
                {
                    ProcessBatches();
                }

                foreach (GradWorker worker in _workers)
                {
                    EngineArray.ArrayCopy(_flat.Weights, 0, worker.Weights, 0, _flat.Weights.Length);
                }

                if (_flat.HasContext)
                {
                    CopyContexts();
                }

                if (_reportedException != null)
                {
                    throw (new EncogError(_reportedException));
                }

                PostIteration();

                EncogLogging.Log(EncogLogging.LevelInfo, "Training iterations done, error: " + Error);
            }
            catch (IndexOutOfRangeException ex)
            {
                EncogValidate.ValidateNetworkForTraining(_network, Training);
                throw new EncogError(ex);
            }
        }

        public double[] LastGradient
        {
            get
            {
                return _lastGradient;
            }
        }

        #region TrainFlatNetwork Members

        public override void FinishTraining()
        {
        }

        public override int IterationNumber
        {
            get
            {
                return _iteration;
            }
            set
            {
                _iteration = value;
            }
        }

        public IContainsFlat Network
        {
            get
            {
                return _network;
            }
        }

        public int NumThreads
        {
            get
            {
                return _numThreads;
            }
            set
            {
                _numThreads = value;
            }
        }

        public override IMLDataSet Training
        {
            get
            {
                return _training;
            }
        }

        #endregion

        public void CalculateGradients()
        {
            if (_workers == null)
            {
                Init();
            }

            if (_flat.HasContext)
            {
                _workers[0].Network.ClearContext();
            }

            _totalError = 0;

            Parallel.ForEach(_workers, worker => worker.Run());

            Error = _totalError / _workers.Length;
        }

        /// <summary>
        /// Copy the contexts to keep them consistent with multithreaded training.
        /// </summary>
        ///
        private void CopyContexts()
        {
            // copy the contexts(layer outputO from each group to the next group
            for (int i = 0; i < (_workers.Length - 1); i++)
            {
                double[] src = _workers[i].Network.LayerOutput;
                double[] dst = _workers[i + 1].Network.LayerOutput;
                EngineArray.ArrayCopy(src, dst);
            }

            // copy the contexts from the final group to the real network
            EngineArray.ArrayCopy(_workers[_workers.Length - 1].Network.LayerOutput, _flat.LayerOutput);
        }

        /// <summary>
        /// Init the process.
        /// </summary>
        ///
        private void Init()
        {
            // fix flat spot, if needed
            _flatSpot = new double[_flat.ActivationFunctions.Length];

            if (FixFlatSpot)
            {
                for (int i = 0; i < _flat.ActivationFunctions.Length; i++)
                {
                    IActivationFunction af = _flat.ActivationFunctions[i];
                    if (af is ActivationSigmoid)
                    {
                        _flatSpot[i] = 0.1;
                    }
                    else
                    {
                        _flatSpot[i] = 0.0;
                    }
                }
            }
            else
            {
                EngineArray.Fill(_flatSpot, 0);
            }

            var determine = new DetermineWorkload(_numThreads, this._indexable.Count);

            _workers = new GradWorker[determine.ThreadCount];

            int index = 0;

            // handle CPU
            foreach (IntRange r in determine.CalculateWorkers())
            {
                _workers[index++] = new GradWorker(
                    ((FlatNetwork)_network.Flat.Clone()),
                    this,
                    _indexable.OpenAdditional(),
                    r.Low,
                    r.High,
                    _flatSpot,
                    ErrorFunction);
            }

            InitOthers();
        }

        internal void Learn()
        {
            double[] weights = _flat.Weights;
            for (int i = 0; i < Gradients.Length; i++)
            {
                weights[i] += UpdateWeight(Gradients, _lastGradient, i);
                Gradients[i] = 0;
            }
        }

        internal void LearnLimited()
        {
            double limit = _flat.ConnectionLimit;
            double[] weights = _flat.Weights;
            for (int i = 0; i < Gradients.Length; i++)
            {
                if (Math.Abs(weights[i]) < limit)
                {
                    weights[i] = 0;
                }
                else
                {
                    weights[i] += UpdateWeight(Gradients, _lastGradient, i);
                }
                Gradients[i] = 0;
            }
        }

        public void Report(double[] gradients, double error, Exception ex)
        {
            lock (this)
            {
                if (ex == null)
                {
                    for (int i = 0; i < gradients.Length; i++)
                    {
                        Gradients[i] += gradients[i];
                    }
                    _totalError += error;
                }
                else
                {
                    _reportedException = ex;
                }
            }
        }

        #endregion

        private void ProcessPureBatch()
        {
            CalculateGradients();

            if (_flat.Limited)
            {
                LearnLimited();
            }
            else
            {
                Learn();
            }
        }

        private void ProcessBatches()
        {
            if (_workers == null)
            {
                Init();
            }

            if (_flat.HasContext)
            {
                _workers[0].Network.ClearContext();
            }

            _workers[0].CalculateError.Reset();

            int lastLearn = 0;

            for (int i = 0; i < Training.Count; i++)
            {
                var randomNumber = rng.Next(Training.Count);
                _workers[0].Run(randomNumber);

                lastLearn++;

                if (lastLearn++ >= BatchSize)
                {
                    if (_flat.Limited)
                    {
                        LearnLimited();
                    }
                    else
                    {
                        Learn();
                        lastLearn = 0;
                    }
                }
            }

            // handle any remaining learning
            if (lastLearn > 0)
            {
                Learn();
            }

            this.Error = _workers[0].CalculateError.Calculate();
        }

        public int BatchSize { get; set; }

        public const String PropertyLastDelta = "LAST_DELTA";

        private double[] _lastDelta;

        private double _learningRate;

        private double _momentum;

        public BackProp(IContainsFlat network, IMLDataSet training, double learnRate, double momentum)
            : base(TrainingImplementationType.Iterative)
        {
            _network = network;
            _flat = network.Flat;
            _training = training;

            Gradients = new double[_flat.Weights.Length];
            _lastGradient = new double[_flat.Weights.Length];

            _indexable = training;
            _numThreads = 0;
            _reportedException = null;
            FixFlatSpot = true;
            ErrorFunction = new LinearErrorFunction();
            rng = new Random(1001);

            ValidateNetwork.ValidateMethodToData(network, training);
            _momentum = momentum;
            _learningRate = learnRate;
            _lastDelta = new double[Network.Flat.Weights.Length];
        }

        public override sealed bool CanContinue
        {
            get
            {
                return true;
            }
        }

        public double[] LastDelta
        {
            get
            {
                return _lastDelta;
            }
        }

        #region ILearningRate Members

        public double LearningRate
        {
            get
            {
                return _learningRate;
            }
            set
            {
                _learningRate = value;
            }
        }

        #endregion

        #region IMomentum Members

        public double Momentum
        {
            get
            {
                return _momentum;
            }
            set
            {
                _momentum = value;
            }
        }

        #endregion

        public bool IsValidResume(TrainingContinuation state)
        {
            if (!state.Contents.ContainsKey(PropertyLastDelta))
            {
                return false;
            }

            if (!state.TrainingType.Equals(GetType().Name))
            {
                return false;
            }

            var d = (double[])state.Get(PropertyLastDelta);
            return d.Length == ((IContainsFlat)Method).Flat.Weights.Length;
        }

        public override sealed TrainingContinuation Pause()
        {
            var result = new TrainingContinuation { TrainingType = GetType().Name };
            result.Set(PropertyLastDelta, _lastDelta);
            return result;
        }

        public override sealed void Resume(TrainingContinuation state)
        {
            if (!IsValidResume(state))
            {
                throw new TrainingError("Invalid training resume data length");
            }

            _lastDelta = (double[])state.Get(PropertyLastDelta);
        }

        public double UpdateWeight(double[] gradients, double[] lastGradient, int index)
        {
            double delta = (gradients[index] * _learningRate) + (_lastDelta[index] * _momentum);
            _lastDelta[index] = delta;
            return delta;
        }

        public void InitOthers()
        {
        }
    }
}