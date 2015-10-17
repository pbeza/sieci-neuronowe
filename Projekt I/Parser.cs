using NDesk.Options;
using System;
using System.Collections.Generic;
using System.IO;

namespace sieci_neuronowe
{
    public class Parser
    {
        public const string DefaultLogFilePath = @".\out.txt";
        public const string DefaultTrainFilePath = @".\data\classification\data.train.csv";
        public const string DefaultTestingFilePath = @".\data\classification\data.test.csv";
        public static readonly string[] DefaultArgs =
        {
            "-" + ShortClassificationOption,
            "-" + ShortLogPathOption,
            DefaultLogFilePath,
            "-" + ShortTestingPathOption,
            DefaultTestingFilePath,
            DefaultTrainFilePath
        };
        public enum ProblemType { Classification, Regression, Unspecified };
        public const string ShortHelpOption = "h",
                             ShortClassificationOption = "c",
                             ShortRegressionOption = "r",
                             ShortTestingPathOption = "t",
                             ShortLogPathOption = "l",
                             LongHelpOption = "help",
                             LongClassificationOption = "classification",
                             LongRegressionOption = "regression",
                             LongTestingPathOption = "testing",
                             LongLogPathOption = "log";
        private const int NumberOfExpectedUnrecognizedOptions = 1;
        public ProblemType Problem { get; private set; }
        public bool InputValid { get; private set; }
        public bool ShowHelpRequested { get; private set; }
        public string LearningSetFilePath { get; private set; }
        public string TestingSetFilePath { get; private set; }
        public string LogFilePath { get; private set; }
        public string MessageForUser { get; private set; }

        public Parser(IEnumerable<string> args)
        {
            Problem = ProblemType.Unspecified;
            InputValid = false;
            ShowHelpRequested = false;
            LearningSetFilePath = string.Empty;
            TestingSetFilePath = string.Empty;
            LogFilePath = string.Empty;
            MessageForUser = string.Empty;
            Parse(args);
        }

        public void PrintUsage(string[] args)
        {
            Console.WriteLine("USAGE:");
            Console.WriteLine();
            Console.WriteLine("  {0} [OPTIONS] LEARNING_SET_PATH", AppDomain.CurrentDomain.FriendlyName);
            Console.WriteLine();
            Console.WriteLine("OPTIONS:");
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1} TESTING_SET_PATH", ShortTestingPathOption, LongTestingPathOption);
            Console.WriteLine("          Path to CSV file with testing set.\n" +
                              "          If not given, testing set is the same as learning set.");
            Console.WriteLine();
            Console.WriteLine("  Exactly one of the following options is required:");
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1}", ShortClassificationOption, LongClassificationOption);
            Console.WriteLine("          Choose classification problem solver.");
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1}", ShortRegressionOption, LongRegressionOption);
            Console.WriteLine("          Choose regression problem solver.");
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
                { ShortLogPathOption + "|" + LongLogPathOption + "=", "Path to log text file for debugging purposes.", v => LogFilePath = v }
            };

            try
            {
                unrecognizedOptions = p.Parse(args); // unrecognized[0] is file path to learning set
            }
            catch (OptionException)
            {
                MessageForUser = "Unexpected arguments.";
            }

            if (unrecognizedOptions.Count < NumberOfExpectedUnrecognizedOptions)
            {
                MessageForUser = "Learning set file path was not specified.";
            }
            else if (unrecognizedOptions.Count > NumberOfExpectedUnrecognizedOptions)
            {
                MessageForUser = "Too many arguments.";
            }
            else if (!File.Exists(LearningSetFilePath = unrecognizedOptions[0]))
            {
                MessageForUser = "Specified learning set file doesn't exist.";
            }

            if (TestingSetFilePath == null)
            {
                TestingSetFilePath = LearningSetFilePath;
            }
            if (LogFilePath == null)
            {
                LogFilePath = DefaultLogFilePath;
            }
            else if (File.Exists(LogFilePath))
            {
                Console.WriteLine("Warning. Log file is going to be overwritten.");
            }

            if (TestingSetFilePath != null && !File.Exists(TestingSetFilePath))
            {
                MessageForUser = "Given testing file doesn't exist.";
            }
            else if (classification && regression)
            {
                MessageForUser = "Both classification and regression flags are set.";
            }
            else if (!classification && !regression)
            {
                MessageForUser = "Neither classification nor regression flag was set.";
            }

            InputValid = MessageForUser == string.Empty;
        }
    }
}