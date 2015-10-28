using NDesk.Options;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace sieci_neuronowe
{
    public class ArgsParser
    {
        public const string DefaultLogFilePath = @".\log.txt";
        public const string DefaultClassificationLearningFilePath = @".\data\samples\classification\data.train.csv";
        public const string DefaultClassificationTestingFilePath = @".\data\samples\classification\data.test.csv";
        public const string DefaultRegressionLearningFilePath = @".\data\samples\regression\data.xsq.train.csv";
        public const string DefaultRegressionTestingFilePath = @".\data\samples\regression\data.xsq.test.csv";
        public const string DefaultNeuralNetworkDefinitionFilePath = @".\data\neural_networks\example.txt";
        public const string DefaultConfigFile = @".\data\config\classification\circles.json";
        public const string ClassificationJsonKey = "classification";
        public const string IterationsJsonKey = "iterations";
        public const string MomentumJsonKey = "momentum";
        public const string LearningRateJsonKey = "learningRate";
        public const string LearningPathJsonKey = "learningPath";
        public const string TestingPathJsonKey = "testingPath";
        public const string NeuralNetworkJsonKey = "networkPath";
        public const int DefaultNumberOfIterations = 1000;
        public const double DefaultMomentumValue = 0.01;
        public const double DefaultLearningRateValue = 0.3;
        public const double MinAllowedMomentumValue = 0.0;
        public const double MaxAllowedMomentumValue = 1.0;
        public const double MinAllowedLearningRate = 0.0;
        public const double MaxAllowedLearningRate = 1.0;
        public static readonly string[] DefaultArgs =
        {
            DefaultConfigFile
        };
        public enum ProblemType { Classification, Regression, Unspecified };
        public const string ShortHelpOption = "h",
                            ShortClassificationOption = "c",
                            ShortRegressionOption = "r",
                            ShortTestingPathOption = "t",
                            ShortLogPathOption = "v",
                            ShortIterationsOption = "n",
                            ShortMomentumValueOption = "l",
                            ShortLearningRateOption = "k",
                            LongHelpOption = "help",
                            LongClassificationOption = "classification",
                            LongRegressionOption = "regression",
                            LongTestingPathOption = "testing",
                            LongLogPathOption = "log",
                            LongIterationsOption = "iterations",
                            LongMomentumValueOption = "momentum",
                            LongLearningRateOption = "rate";
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
        public double LearningRate { get; private set; }
        public string MessageForUser { get; private set; }

        public ArgsParser(IList<string> args)
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
            LearningRate = DefaultLearningRateValue;
            MessageForUser = string.Empty;
            if (args.Count == 1 && (args[0] == "--" + LongHelpOption || args[0] == "-" + ShortHelpOption))
            {
                ShowHelpRequested = true;
                InputValid = true;
            }
            else
            {
                var stringToParse = args.Count == 1 ? JsonConfigFileToStringOptions(args[0]) : args;
                if (stringToParse == null) return;
                ParseCmdLineOptions(stringToParse.ToArray());
            }
        }

        public void PrintUsage(string[] args)
        {
            const string learningSetPath = "LEARNING_SET_PATH",
                         networkDefinitionPath = "NETWORK_DEF_PATH",
                         jsonConfigPath = "JSON_CONFIG_PATH";
            Console.WriteLine("USAGE:");
            Console.WriteLine();
            Console.WriteLine("  {0} [-{1}|-{2}] [OPTIONS] {3} {4}",
                              AppDomain.CurrentDomain.FriendlyName,
                              ShortClassificationOption,
                              ShortRegressionOption,
                              learningSetPath,
                              networkDefinitionPath);
            Console.WriteLine("  {0} {1}", AppDomain.CurrentDomain.FriendlyName, jsonConfigPath);
            Console.WriteLine();
            Console.WriteLine("WHERE:");
            Console.WriteLine();
            Console.WriteLine("    {0,-24} is path to CSV file with learning set.", learningSetPath);
            Console.WriteLine();
            Console.WriteLine("    {0,-24} is path to text file with neural network definition.", networkDefinitionPath);
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1}", ShortClassificationOption, LongClassificationOption);
            Console.WriteLine("          Choose classification problem solver.");
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1}", ShortRegressionOption, LongRegressionOption);
            Console.WriteLine("          Choose regression problem solver.");
            Console.WriteLine();
            Console.WriteLine("    {0,-24} is path to JSON configuration file path.", jsonConfigPath);
            Console.WriteLine();
            Console.WriteLine("OPTIONS:");
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1} PATH", ShortTestingPathOption, LongTestingPathOption);
            Console.WriteLine("          Path to CSV file with testing set.");
            Console.WriteLine("          If not given, testing set is the same as learning set.");
            Console.WriteLine("          If not specified, default path PATH={0} is assigned.", DefaultClassificationTestingFilePath);
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1} N", ShortIterationsOption, LongIterationsOption);
            Console.WriteLine("          Number of iterations for learning process.");
            Console.WriteLine("          If not specified, default number N={0} is assigned.", DefaultNumberOfIterations);
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1} VAL", ShortMomentumValueOption, LongMomentumValueOption);
            Console.WriteLine("          Momentum value used for learning process. VAL must be from range [{0}; {1}].", MinAllowedMomentumValue, MaxAllowedMomentumValue);
            Console.WriteLine("          If not specified, default momentum value VAL={0} is assigned.", DefaultMomentumValue);
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1} VAL", ShortLearningRateOption, LearningRate);
            Console.WriteLine("          Learning rate. Must be value from range [{0}; {1}].", MinAllowedLearningRate, MaxAllowedLearningRate);
            Console.WriteLine("          If not specified, default learning rate value VAL={0} is assigned.", DefaultLearningRateValue);
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1} PATH", ShortLogPathOption, LongLogPathOption);
            Console.WriteLine("          Path to log text file which will be created.");
            Console.WriteLine("          If not specified, default path PATH={0} is assigned.", DefaultLogFilePath);
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1}", ShortHelpOption, LongHelpOption);
            Console.WriteLine("          Print this help/usage and exit.");
        }

        private IList<string> JsonConfigFileToStringOptions(string path)
        {
            if (!File.Exists(path))
            {
                InputValid = false;
                MessageForUser = string.Format("Config file {0} does NOT exist.", path);
                return null;
            }
            var configString = File.ReadAllText(path);
            var dict = JsonConvert.DeserializeObject<Dictionary<string, string>>(configString);
            string classificationString = string.Empty,
                   iterationsString = string.Empty,
                   momentumString = string.Empty,
                   learningRate = string.Empty,
                   learningPath = string.Empty,
                   testingPath = string.Empty,
                   networkPath = string.Empty;
            if (!dict.TryGetValue(ClassificationJsonKey, out classificationString))
            {
                SetMessageKeyNotFound(ClassificationJsonKey, path);
            }
            else if (!dict.TryGetValue(IterationsJsonKey, out iterationsString))
            {
                SetMessageKeyNotFound(IterationsJsonKey, path);
            }
            else if (!dict.TryGetValue(MomentumJsonKey, out momentumString))
            {
                SetMessageKeyNotFound(MomentumJsonKey, path);
            }
            else if (!dict.TryGetValue(LearningRateJsonKey, out learningRate))
            {
                SetMessageKeyNotFound(LearningRateJsonKey, path);
            }
            else if (!dict.TryGetValue(LearningPathJsonKey, out learningPath))
            {
                SetMessageKeyNotFound(LearningPathJsonKey, path);
            }
            else if (!dict.TryGetValue(TestingPathJsonKey, out testingPath))
            {
                SetMessageKeyNotFound(TestingPathJsonKey, path);
            }
            else if (!dict.TryGetValue(NeuralNetworkJsonKey, out networkPath))
            {
                SetMessageKeyNotFound(NeuralNetworkJsonKey, path);
            }
            if (MessageForUser != string.Empty)
            {
                InputValid = false;
                return null;
            }
            bool isClassification;
            bool.TryParse(classificationString, out isClassification);
            return new List<string>
            {
                "-" + (isClassification ? ShortClassificationOption : ShortRegressionOption),
                "-" + ShortIterationsOption, iterationsString,
                "-" + ShortMomentumValueOption, momentumString.Replace('.', ','), // locale issue
                "-" + ShortLearningRateOption, learningRate.Replace('.', ','), // locale issue
                "-" + ShortTestingPathOption, testingPath,
                learningPath,
                networkPath
            };
        }

        private void SetMessageKeyNotFound(string keyName, string path)
        {
            MessageForUser = string.Format("Key '{0}' was not found in '{1}' configuration file.", keyName, path);
        }

        private void ParseCmdLineOptions(IEnumerable<string> args)
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
                { ShortLearningRateOption + "|" + LongLearningRateOption + "=", "Learning rate.", (double v) => LearningRate = v },
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
            else if (LearningRate < MinAllowedLearningRate || LearningRate > MaxAllowedLearningRate)
            {
                MessageForUser = string.Format("Learning rate value must be from range [{0}; {1}].", MinAllowedLearningRate, MaxAllowedLearningRate);
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