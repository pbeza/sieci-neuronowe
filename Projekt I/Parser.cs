using System;
using System.Collections.Generic;
using System.IO;
using NDesk.Options;

namespace sieci_neuronowe
{
    internal class Parser
    {
        public enum ProblemType { Classification, Regression };
        private const string ShortHelpOption = "h",
                             ShortQlassificationOption = "c",
                             ShortRegressionOption = "r",
                             ShortTestingPathOption = "t",
                             LongHelpOption = "help",
                             LongQlassificationOption = "classification",
                             LongRegressionOption = "regression",
                             LongTestingPathOption = "testing";
        public ProblemType Problem { get; protected set; }
        public string InputFilePath { get; protected set; }
        public string TestingFilePath { get; protected set; }

        public Parser(string[] args)
        {
            var showHelp = false;
            var unrecognized = new List<string>();
            var classification = false;
            var regression = false;
            var p = new OptionSet {
                { ShortHelpOption + "|" + LongHelpOption, "Print usage.", v => showHelp = v != null },
                { ShortQlassificationOption + "|" + LongQlassificationOption, "Choose classification problem.", v => { if (v != null) classification = true; } },
                { ShortRegressionOption + "|" + LongRegressionOption, "Choose regression problem.", v => { if (v != null) regression = true; } },
                { ShortTestingPathOption + "|" + LongTestingPathOption + "=", "Path to testing CSV.", v => TestingFilePath = v }
            };

            try
            {
                unrecognized = p.Parse(args); // unrecognized[0] is file path to learning set
            }
            catch (OptionException)
            {
                showHelp = true;
            }

            if (showHelp || unrecognized.Count != 1 || !File.Exists(unrecognized[0]) ||
                (TestingFilePath != null && !File.Exists(TestingFilePath)) ||
                (regression && classification) || (!regression && !classification))
            {
                PrintUsage(args);
                Environment.Exit(1);
            }
            else
            {
                InputFilePath = unrecognized[0];
            }
        }

        protected void PrintUsage(string[] args)
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
            Console.WriteLine("    -{0}, --{1}", ShortQlassificationOption, LongQlassificationOption);
            Console.WriteLine("          Choose classification problem solver.");
            Console.WriteLine();
            Console.WriteLine("    -{0}, --{1}", ShortRegressionOption, LongRegressionOption);
            Console.WriteLine("          Choose regression problem solver.");
        }
    }
}