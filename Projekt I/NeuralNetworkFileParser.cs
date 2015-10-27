using Encog.Engine.Network.Activation;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace sieci_neuronowe
{
    using System.Globalization;

    public class NeuralNetworkFileParser
    {
        private const char TextSeparator = ' ';
        private const char CharacterStartingComment = '#';
        private const char BiasMarkCharacterPostfix = 'b';
        private const int MinTotalNumberOfLayers = 2; // input + outplut layer (0 hidden layers)
        private readonly IEnumerator<string> _fileContentEnumerator;
        private readonly BasicNetwork _neuralNetwork = new BasicNetwork();
        private readonly IList<IActivationFunction> _layersActivationFunctions = new List<IActivationFunction>();
        private readonly IList<int> _neuronsInLayers = new List<int>();
        private readonly ISet<int> _indicesOfBiasDefinedLayers = new HashSet<int>();
        private IList<double> _layersBiases = new List<double>();
        private string _currentLine;
        private int _currentLineNumber;

        private readonly Random random;

        private int TotalLayersNumber { get { return _neuronsInLayers.Count; } }

        private enum ActivationFunctionsNames
        {
            sigmoid,
            linear,
            bipolar,
            sin,
            tanh,
            elliott,
            elliott_sym,
            gaussian,
            bipolar_sigmoid,
            off
        };

        public NeuralNetworkFileParser(string path)
        {
            _fileContentEnumerator = File.ReadLines(path).GetEnumerator();
            random = new Random();
        }

        public BasicNetwork Parse()
        {
            // Fetch first not empty line (which should be sequence of numbers
            // of neurons in subsequent layers), set number of neurons for
            // all the layers and save layers' numbers for which bias is defined.

            SetListOfNeuronsInAllLayers();

            // Fetch first not empty line (which should be sequence of layers' activation names)
            // and set: activation functions for all of the layers.

            SetActivationFunctionsForAllLayers();

            // Read and save biases for 'b'-marked network layers

            SetListOfLayersBiases();

            // Add all layers to constructed neural network

            AddAllLayersToNeuralNetwork();

            // Add biases to neural network

            AddAllBiasesToNeuralNetwork();

            // Add all weights to constructed neural network

            return AddAllWeightsToConstructedNeuralNetwork();
        }

        private void SetListOfNeuronsInAllLayers()
        {
            SetFirstNotIgnoredLineAsCurrentLine("File is empty.");
            var currentLayerIndex = 0;
            foreach (var s in _currentLine.Split(TextSeparator))
            {
                var n = s;
                if (s.Last() == BiasMarkCharacterPostfix)
                {
                    n = s.TrimEnd(BiasMarkCharacterPostfix);
                    _indicesOfBiasDefinedLayers.Add(currentLayerIndex);
                }
                int numberOfNeurons;
                if (!int.TryParse(n, out numberOfNeurons))
                {
                    ThrowErrorFileLoadFile("Given line contains not a number value.");
                }
                if (numberOfNeurons <= 0)
                {
                    ThrowErrorFileLoadFile("Number of neurons in all layers must be positive.");
                }
                _neuronsInLayers.Add(numberOfNeurons);
                currentLayerIndex++;
            }
            if (TotalLayersNumber < MinTotalNumberOfLayers)
            {
                var msg = string.Format("Minimal number of layers is {0} (specified {1} layer(s)).", MinTotalNumberOfLayers, TotalLayersNumber);
                ThrowErrorFileLoadFile(msg);
            }
        }

        private void SetActivationFunctionsForAllLayers()
        {
            SetFirstNotIgnoredLineAsCurrentLine("Unexpected end of file. List of activation functions' names expected.");
            SetActivationFunctionsForAllLayersFromLine();
        }

        private void SetActivationFunctionsForAllLayersFromLine()
        {
            var splitted = _currentLine.Split();
            if (splitted.Length != TotalLayersNumber)
            {
                var msg = string.Format("Number of all layers is {0} and number of activation functions' names is {1}. They must be equal.", splitted.Length, TotalLayersNumber);
                ThrowErrorFileLoadFile(msg);
            }
            foreach (var activationFunctionName in splitted)
            {
                _layersActivationFunctions.Add(CreateActivationFunctionFromString(activationFunctionName));
            }
        }

        private void SetListOfLayersBiases()
        {
            if (_indicesOfBiasDefinedLayers.Count == 0)
            {
                return;
            }

            SetFirstNotIgnoredLineAsCurrentLine("Unexpected end of file. File does NOT specify layers' biases.");
            SetBiasesForEveryNetworkLayerFromCurrentLine();
        }

        private void SetBiasesForEveryNetworkLayerFromCurrentLine()
        {
            _layersBiases = Array.ConvertAll(_currentLine.Split(TextSeparator), double.Parse);
            if (_layersBiases.Count != _indicesOfBiasDefinedLayers.Count)
            {
                var msg = string.Format("Number of specified biases is {0}. It should be equal to number of '{1}'-marked layers from 1st section of the file (which is {2}).",
                                        _layersBiases.Count, BiasMarkCharacterPostfix, _indicesOfBiasDefinedLayers.Count);
                ThrowErrorFileLoadFile(msg);
            }
        }

        private BasicNetwork AddAllWeightsToConstructedNeuralNetwork()
        {

            // layer no. 0 is input layer
            int processedLayers = 0,
                processedNeuronsWithinLayer = 0,
                processedNotIgnoredLines = 0,
                numberOfNeuronsInCurrentLayer = _neuronsInLayers[processedLayers],
                numberOfNeuronsInNextLayer = _neuronsInLayers[processedLayers + 1];

            while (true)
            {
                SetFirstNotIgnoredLineAsCurrentLine();
                if (_currentLine == null)
                    break; // EOF
                processedNotIgnoredLines++;
                if (processedLayers >= TotalLayersNumber - 1)
                {
                    var msg = string.Format("Too many neural layers. Number of processed layers is {0}. Declared number of layers is {1}.", processedLayers, TotalLayersNumber);
                    ThrowErrorFileLoadFile(msg);
                }

                if (processedLayers == 0 && _currentLine.Split(',').Length == _neuralNetwork.EncodedArrayLength())
                {
                    Console.WriteLine("Found weight dump. Decoding {0} weights.", _neuralNetwork.EncodedArrayLength());
                    var dumpedWeights = Array.ConvertAll(
                        _currentLine.Split(','),
                        (input => double.Parse(input, CultureInfo.InvariantCulture)));
                    _neuralNetwork.DecodeFromArray(dumpedWeights);
                    return _neuralNetwork;
                }

                var weightsForOneVertex = GetWeightsFromCurrentLine(numberOfNeuronsInNextLayer);
                if (weightsForOneVertex.Length == 0)
                {
                    weightsForOneVertex = Enumerable.Repeat(1.0, numberOfNeuronsInNextLayer).ToArray();
                }

                if (weightsForOneVertex.Length != numberOfNeuronsInNextLayer)
                {
                    var msg = string.Format(
                        "Number of weights for layer no. {0} should equal {1} (is {2}).",
                        processedLayers,
                        numberOfNeuronsInNextLayer,
                        weightsForOneVertex.Length);
                    ThrowErrorFileLoadFile(msg);
                }
                for (var toNeuronNumber = 0; toNeuronNumber < numberOfNeuronsInNextLayer; toNeuronNumber++)
                {
                    _neuralNetwork.SetWeight(processedLayers, processedNeuronsWithinLayer, toNeuronNumber, weightsForOneVertex[toNeuronNumber]);
                }
                if (++processedNeuronsWithinLayer != numberOfNeuronsInCurrentLayer) continue;
                if (++processedLayers == TotalLayersNumber - 1)
                    break;
                processedNeuronsWithinLayer = 0;
                // Bias is treated like a neuron in layer below
                numberOfNeuronsInCurrentLayer = _neuronsInLayers[processedLayers];
                numberOfNeuronsInNextLayer = _neuronsInLayers[processedLayers + 1];
            }

            // Check if file ended too early
            if (processedNotIgnoredLines != GetTotalNumberOfExpectedNotIgnoredLines() || processedLayers != TotalLayersNumber - 1)
                ThrowErrorFileLoadFile("Unexpected end of file. Wrong file format.");

            return _neuralNetwork;
        }

        private void AddAllLayersToNeuralNetwork()
        {
            _neuralNetwork.AddLayer(new BasicLayer(_layersActivationFunctions.First(), _indicesOfBiasDefinedLayers.Contains(0), _neuronsInLayers.First()));
            for (var i = 1; i < _neuronsInLayers.Count - 1; i++)
            {
                _neuralNetwork.AddLayer(new BasicLayer(_layersActivationFunctions[i], _indicesOfBiasDefinedLayers.Contains(i), _neuronsInLayers[i]));
            }
            _neuralNetwork.AddLayer(new BasicLayer(_layersActivationFunctions.Last(), _indicesOfBiasDefinedLayers.Contains(_neuronsInLayers.Count - 1), _neuronsInLayers.Last()));
            _neuralNetwork.Structure.FinalizeStructure();
        }

        private void AddAllBiasesToNeuralNetwork()
        {
            for (int i = 0, j = 0; i < _layersBiases.Count; i++)
            {
                if (_indicesOfBiasDefinedLayers.Contains(i))
                {
                    _neuralNetwork.SetLayerBiasActivation(i, _layersBiases[j++]);
                }
            }
        }

        private int GetTotalNumberOfExpectedNotIgnoredLines()
        {
            var totalNumberOfWeights = 0;
            for (var i = 0; i < _neuronsInLayers.Count - 1; i++)
            {
                totalNumberOfWeights += _neuronsInLayers[i];
            }
            return totalNumberOfWeights;
        }

        private double[] GetWeightsFromCurrentLine(int expectedNumber)
        {
            var weights = new double[expectedNumber];
            var splitLine = _currentLine.Split(TextSeparator);
            bool ok = splitLine.Length == expectedNumber;
            for (int index = 0; index < splitLine.Length; index++)
            {
                if (!double.TryParse(splitLine[index], out weights[index]))
                {
                    weights[index] = (this.random.NextDouble() * 2.0)-1.0;
                    ok = false;
                }
            }

            if (!ok)
            {
                var msg = string.Format("Expected {0} weights, got {1}", expectedNumber, splitLine.Length);
                ThrowErrorFileLoadFile(msg);
            }

            return weights;
        }

        private void SetFirstNotIgnoredLineAsCurrentLine(string msgIfNoMoreLines = null)
        {
            while (_fileContentEnumerator.MoveNext())
            {
                _currentLineNumber++;
                _currentLine = RemoveIgnoredSubstringsFromString(_fileContentEnumerator.Current);
                if (_currentLine != string.Empty)
                    break;
            }
            if (msgIfNoMoreLines != null && _currentLine == null)
            {
                ThrowErrorFileLoadFile(msgIfNoMoreLines);
            }
        }

        private static string RemoveIgnoredSubstringsFromString(string line)
        {
            var index = line.IndexOf(CharacterStartingComment);
            if (index != -1)
                line = line.Remove(index); // remove comment
            line = line.Trim();
            return Regex.Replace(line, @"\s+", " ");
        }

        private void ThrowErrorFileLoadFile(string errorMsg, bool critical = false)
        {
            var msgPrefix = string.Format("Error in line {0} reading neural network file. Make sure you have correct file's format. ", _currentLineNumber);
            if (critical)
            {
                throw new FileLoadException(msgPrefix + errorMsg);
            }

            Console.WriteLine(msgPrefix + errorMsg);
        }

        private IActivationFunction CreateActivationFunctionFromString(string activationFunctionName)
        {
            ActivationFunctionsNames parsedEnum;
            if (!Enum.TryParse(activationFunctionName, out parsedEnum))
            {
                var msg = string.Format("Unrecognized activation function: '{0}'.", activationFunctionName);
                ThrowErrorFileLoadFile(msg);
            }
            switch (parsedEnum)
            {
                case ActivationFunctionsNames.bipolar:
                    return new ActivationBiPolar();
                case ActivationFunctionsNames.linear:
                    return new ActivationLinear();
                case ActivationFunctionsNames.sigmoid:
                    return new ActivationSigmoid();
                case ActivationFunctionsNames.sin:
                    return new ActivationSIN();
                case ActivationFunctionsNames.tanh:
                    return new ActivationTANH();
                case ActivationFunctionsNames.elliott:
                    return new ActivationElliott();
                case ActivationFunctionsNames.elliott_sym:
                    return new ActivationElliottSymmetric();
                case ActivationFunctionsNames.gaussian:
                    return new ActivationGaussian();
                case ActivationFunctionsNames.bipolar_sigmoid:
                    return new ActivationBipolarSteepenedSigmoid();
                case ActivationFunctionsNames.off:
                    return null;
            }

            throw new NotImplementedException();
        }
    }
}
