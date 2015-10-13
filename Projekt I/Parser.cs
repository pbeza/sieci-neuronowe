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
                             ShortQlassificationOption = "q",
                             ShortRegressionOption = "r",
                             LongHelpOption = "help",
                             LongQlassificationOption = "classification",
                             LongRegressionOption = "regression";
        public ProblemType Problem { get; protected set; }
        public string InputFilePath { get; protected set; }
        public bool Valid { get; protected set; }

        public Parser(string[] args)
        {
            var showHelp = false;
            var unrecognized = new List<string>();
            var p = new OptionSet {
                { ShortHelpOption + "|" + LongHelpOption, "Print usage.", v => showHelp = v != null },
                { ShortQlassificationOption + "|" + LongQlassificationOption, "Choose classification problem.",
                    v => { if (v != null) Problem = ProblemType.Classification; } },
                { ShortRegressionOption + "|" + LongRegressionOption, "Choose regression problem.",
                    v => { if (v != null) Problem = ProblemType.Regression; } }
            };

            try
            {
                unrecognized = p.Parse(args);
            }
            catch (OptionException)
            {
                showHelp = true;
            }

            if (showHelp || unrecognized.Count != 1 || !File.Exists(unrecognized[0]))
            {
                PrintUsage(args);
                Valid = false;
            }
            else
            {
                InputFilePath = unrecognized[0];
                Valid = true;
            }
        }

        protected void PrintUsage(string[] args)
        {
            Console.WriteLine("USAGE:");
            Console.WriteLine();
            Console.WriteLine("  {0} [OPTIONS] FILE.CSV", AppDomain.CurrentDomain.FriendlyName);
            Console.WriteLine();
            Console.WriteLine("OPTIONS:");
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