using NDesk.Options;
using System;
using System.Collections.Generic;
using System.IO;

namespace sieci_neuronowe
{
    public class CommandLineParser
    {
        public const string DefaultLogFilePath = @".\out.txt";
        public const string DefaultLearningFilePath = @".\data\classification\data.train.csv";
        public const string DefaultTestingFilePath = @".\data\classification\data.test.csv";
        public const string DefaultRegressionLearningFilePath = @".\data\regression\data.xsq.train.csv";
        public const string DefaultRegressionTestingFilePath = @".\data\regression\data.xsq.test.csv";
        public const string DefaultNeuralNetworkDefinitionFilePath = @".\data\sample_neural_networks\format_presentation_of_neural_network.txt";
        public const int DefaultNumberOfIterations = 10000;
        public const double DefaultMomentumValue = 0.01;
        public const double MinAllowedMomentumValue = 0.0;
        public const double MaxAllowedMomentumValue = 1.0;
        public static readonly string[] DefaultArgs =
        {
            "-" + ShortClassificationOption,
            "-" + ShortTestingPathOption, DefaultTestingFilePath,
            DefaultLearningFilePath,
            DefaultNeuralNetworkDefinitionFilePath
        };
        public enum ProblemType { Classification, Regression, Unspecified };
        public const string ShortHelpOption = "h",
                            ShortClassificationOption = "c",
                            ShortRegressionOption = "r",
                            ShortTestingPathOption = "t",
                            ShortLogPathOption = "l",
                            ShortIterationsOption = "n",
                            ShortMomentumValueOption = "i",
                            LongHelpOption = "help",
                            LongClassificationOption = "classification",
                            LongRegressionOption = "regression",
                            LongTestingPathOption = "testing",
                            LongLogPathOption = "log",
                            LongIterationsOption = "iterations",
                            LongMomentumValueOption = "inertia";
        private const int NumberOfExpectedUnrecognizedOptions = 2;
        public ProblemType Problem { get; private set; }
        public bool InputValid { get; private set; }
        public bool ShowHelpRequested { get; private set; }
        public string LearningSetFilePath { get; private set; }
        public string TestingSetFilePath { get; private set; }
        public string LogFilePath { get; private set; }
        public string NeuralNetworkDefinitionFilePath { get; private set; }
        public int NumberOfIterations { get; private set; }
        public double Momentum { get; private set; }
        public string MessageForUser { get; private set; }

        public CommandLineParser(IEnumerable<string> args)
        {
            Problem = ProblemType.Unspecified;
            InputValid = false;
            ShowHelpRequested = false;
            LearningSetFilePath = string.Empty;
            TestingSetFilePath = string.Empty;
            LogFilePath = string.Empty;
            NeuralNetworkDefinitionFilePath = string.Empty;
            NumberOfIterations = DefaultNumberOfIterations;
            Momentum = DefaultMomentumValue;
            MessageForUser = string.Empty;
            Parse(args);
        }

        public void PrintUsage(string[] args)
        {
            const string LearningSetPath = "LEARNING_SET_PATH",
                         NetworkDefinitionPath = "NETWORK_DEFINITION_PATH";
            Console.WriteLine("USAGE:");
            Console.WriteLine();
            Console.WriteLine("  {0} [-{1}|-{2}] [OPTIONS] {3} {4}",
                              AppDomain.CurrentDomain.FriendlyName,
                              ShortClassificationOption,
                              ShortRegressionOption,
                              LearningSetPath,
                              NetworkDefinitionPath);
            Console.WriteLine();
            Console.WriteLine("WHERE:");
            Console.WriteLine();
            Console.WriteLine("    {0,-24} is path to CSV file with learning set.", LearningSetPath);
            Console.WriteLine();
            Console.WriteLine("    {0,-24} is path to text file with neural network definition.", NetworkDefinitionPath);
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1}", ShortClassificationOption, LongClassificationOption);
            Console.WriteLine("          Choose classification problem solver.");
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1}", ShortRegressionOption, LongRegressionOption);
            Console.WriteLine("          Choose regression problem solver.");
            Console.WriteLine();
            Console.WriteLine("OPTIONS:");
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1} PATH", ShortTestingPathOption, LongTestingPathOption);
            Console.WriteLine("          Path to CSV file with testing set.");
            Console.WriteLine("          If not given, testing set is the same as learning set.");
            Console.WriteLine("          If not specified, default path PATH={0} is assigned.", DefaultTestingFilePath);
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1} N", ShortIterationsOption, LongIterationsOption);
            Console.WriteLine("          Number of iterations for learning process.");
            Console.WriteLine("          If not specified, default number N={0} is assigned.", DefaultNumberOfIterations);
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1} VAL", ShortMomentumValueOption, LongMomentumValueOption);
            Console.WriteLine("          Momentum value used for learning process. VAL must be from range [{0}; {1}].", MinAllowedMomentumValue, MaxAllowedMomentumValue);
            Console.WriteLine("          If not specified, default inertia value VAL={0} is assigned.", DefaultMomentumValue);
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1} PATH", ShortLogPathOption, LongLogPathOption);
            Console.WriteLine("          Path to log text file which will be created.");
            Console.WriteLine("          If not specified, default path PATH={0} is assigned.", DefaultLogFilePath);
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1}", ShortHelpOption, LongHelpOption);
            Console.WriteLine("          Print this help/usage and exit.");
        }

        private void Parse(IEnumerable<string> args)
        {
            var unrecognizedOptions = new List<string>();
            var classification = false;
            var regression = false;
            var p = new OptionSet {
                { ShortHelpOption + "|" + LongHelpOption, "Print usage.", v => ShowHelpRequested = v != null },
                { ShortClassificationOption + "|" + LongClassificationOption, "Choose classification problem.", v => { if (v != null) classification = true; } },
                { ShortRegressionOption + "|" + LongRegressionOption, "Choose regression problem.", v => { if (v != null) regression = true; } },
                { ShortTestingPathOption + "|" + LongTestingPathOption + "=", "Path to testing CSV.", v => TestingSetFilePath = v },
                { ShortIterationsOption + "|" + LongIterationsOption + "=", "Number of iterations.", (int v) => NumberOfIterations = v },
                { ShortMomentumValueOption + "|" + LongMomentumValueOption + "=", "Momentum value for learning process.", (double v) => Momentum = v },
                { ShortLogPathOption + "|" + LongLogPathOption + "=", "Path to log text file for debugging purposes.", v => LogFilePath = v }
            };

            try
            {
                unrecognizedOptions = p.Parse(args); // unrecognized[0] is file path to learning set
            }
            catch (OptionException e)
            {
                MessageForUser = e.Message;
            }

            if (ShowHelpRequested)
            {
                SetParserState();
                return;
            }

            if (unrecognizedOptions.Count < NumberOfExpectedUnrecognizedOptions)
            {
                MessageForUser = "Learning set file path or neural network definition file was not specified.";
            }
            else if (unrecognizedOptions.Count > NumberOfExpectedUnrecognizedOptions)
            {
                MessageForUser = "Too many arguments.";
            }
            else if (NumberOfIterations < 0)
            {
                MessageForUser = "Number of iterations must be positive number";
            }
            else if (Momentum < MinAllowedMomentumValue || Momentum > MaxAllowedMomentumValue)
            {
                MessageForUser = string.Format("Momentum value must be from range [{0}; {1}].", MinAllowedMomentumValue, MaxAllowedMomentumValue);
            }
            else if (!File.Exists(LearningSetFilePath = unrecognizedOptions[0]))
            {
                MessageForUser = "Specified file with learning set doesn't exist.";
            }
            else if (!File.Exists(NeuralNetworkDefinitionFilePath = unrecognizedOptions[1]))
            {
                MessageForUser = "Specified file with neural network definition doesn't exist.";
            }

            if (MessageForUser != string.Empty)
            {
                SetParserState();
                return;
            }

            if (string.IsNullOrEmpty(TestingSetFilePath))
            {
                Console.WriteLine("Testing set was not specified explicity. Assuming that learning set is the same as testing set.");
                TestingSetFilePath = LearningSetFilePath;
            }
            else if (!File.Exists(TestingSetFilePath))
            {
                MessageForUser = "Given testing file doesn't exist.";
            }

            if (MessageForUser != string.Empty)
            {
                SetParserState();
                return;
            }

            if (string.IsNullOrEmpty(LogFilePath))
            {
                Console.WriteLine("Log file path was not specified explicity. Assuming that log file path is '{0}'.", DefaultLogFilePath);
                LogFilePath = DefaultLogFilePath;
            }
            else if (File.Exists(LogFilePath))
            {
                Console.WriteLine("Warning! Log file " + LogFilePath + " is going to be overwritten.");
            }

            if (classification && regression)
            {
                MessageForUser = "Both classification and regression flags are set.";
            }
            else if (!classification && !regression)
            {
                MessageForUser = "Neither classification nor regression flag was set.";
            }

            Problem = classification ? ProblemType.Classification : ProblemType.Regression;

            SetParserState();
        }

        private void SetParserState()
        {
            InputValid = MessageForUser == string.Empty;
            if (!InputValid)
            {
                MessageForUser = string.Format("Error. {0}\n\nTry '.\\{1} --{2}' for more help.", MessageForUser, AppDomain.CurrentDomain.FriendlyName, LongHelpOption);
            }
        }
    }
}