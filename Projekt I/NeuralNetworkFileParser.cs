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
    public class NeuralNetworkFileParser
    {
        private const char TextSeparator = ' ';
        private const char CharacterStartingComment = '#';
        private const string ExceptionPostfix = "Error reading neural network file. Make sure you have correct file's format. ";
        private const int MinTotalNumberOfLayers = 2; // input + outplut layer (0 hidden layers)
        private readonly IEnumerator<string> _fileContentEnumerator;
        private readonly BasicNetwork _neuralNetwork = new BasicNetwork();
        private readonly IList<IActivationFunction> _layersActivationFunctions = new List<IActivationFunction>();
        private IList<int> _neuronsInLayers = new List<int>();
        private IList<double> _layersBiases = new List<double>();
        private string _currentLine;
        private int _currentLineNumber;
        private int TotalLayersNumber { get { return _neuronsInLayers.Count; } }
        private enum ActivationFunctionsNames { sigmoid, linear, bipolar, sinus, off };

        public NeuralNetworkFileParser(string path)
        {
            _fileContentEnumerator = File.ReadLines(path).GetEnumerator();
        }

        public BasicNetwork Parse()
        {
            // Fetch first not empty line (which should be sequence of numbers
            // of neurons in subsequent layers) and set number of neurons for all the layers.

            SetListOfNeuronsInAllLayers();

            // Fetch first not empty line (which should be sequence of layers' activation names)
            // and set: activation function for all of the layers.

            SetActivationFunctionsForAllLayers();

            // Read and save bias for every network layer

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
            SetNeuronsInAllLayersFromCurrentLine();
            if (TotalLayersNumber < MinTotalNumberOfLayers)
            {
                ThrowErrorFileLoadFile("Minimal number of layers is " + MinTotalNumberOfLayers + " (specified " + TotalLayersNumber + " layer(s)).");
            }
        }

        private void SetNeuronsInAllLayersFromCurrentLine()
        {
            _neuronsInLayers = Array.ConvertAll(_currentLine.Split(TextSeparator), int.Parse);
            if (_neuronsInLayers.Any(neuronsInLayer => neuronsInLayer <= 0))
            {
                ThrowErrorFileLoadFile("One of the specified layers has less than 1 neuron.");
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
                var msg = string.Format("Number of all layers is {0} and number of activation functions' names is {1}. They must be equal.", TotalLayersNumber, TotalLayersNumber);
                ThrowErrorFileLoadFile(msg);
            }
            foreach (var activationFunctionName in splitted)
            {
                _layersActivationFunctions.Add(CreateActivationFunctionFromString(activationFunctionName));
            }
        }

        private void SetListOfLayersBiases()
        {
            SetFirstNotIgnoredLineAsCurrentLine("Unexpected end of file. File does NOT specify layers' biases.");
            SetBiasesForEveryNetworkLayerFromCurrentLine();
        }

        private void SetBiasesForEveryNetworkLayerFromCurrentLine()
        {
            _layersBiases = Array.ConvertAll(_currentLine.Split(TextSeparator), double.Parse);
            if (_layersBiases.Count != TotalLayersNumber)
            {
                var msg = string.Format("Number of specified biases is {0}. It should be equal to number of all of the layers which is {1}.", _layersBiases.Count, TotalLayersNumber);
                ThrowErrorFileLoadFile(msg);
            }
        }

        private BasicNetwork AddAllWeightsToConstructedNeuralNetwork()
        {
            int processedLayers = 0, // layer no. 0 is input layer
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
                var weightsForOneVertex = GetWeightsFromCurrentLine();
                if (weightsForOneVertex.Length != numberOfNeuronsInNextLayer)
                {
                    var msg = string.Format("Number of weights for layer no. {0} should equal {1} (is {2}).",
                                            processedLayers, numberOfNeuronsInNextLayer, weightsForOneVertex.Length);
                    ThrowErrorFileLoadFile(msg);
                }
                for (var toNeuronNumber = 0; toNeuronNumber < numberOfNeuronsInNextLayer; toNeuronNumber++)
                {
                    _neuralNetwork.AddWeight(processedLayers, processedNeuronsWithinLayer, toNeuronNumber, weightsForOneVertex[toNeuronNumber]);
                }
                if (++processedNeuronsWithinLayer != numberOfNeuronsInCurrentLayer) continue;
                if (++processedLayers == TotalLayersNumber - 1)
                    break;
                processedNeuronsWithinLayer = 0;
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
            const bool isBiasSet = true; // TODO
            _neuralNetwork.AddLayer(new BasicLayer(_layersActivationFunctions.First(), isBiasSet, _neuronsInLayers.First()));
            for (var i = 1; i < _neuronsInLayers.Count - 1; i++)
            {
                _neuralNetwork.AddLayer(new BasicLayer(_layersActivationFunctions[i], isBiasSet, _neuronsInLayers[i]));
            }
            _neuralNetwork.AddLayer(new BasicLayer(_layersActivationFunctions.Last(), isBiasSet, _neuronsInLayers.Last()));
            _neuralNetwork.Structure.FinalizeStructure();
        }

        private void AddAllBiasesToNeuralNetwork()
        {
            for (var i = 0; i < _layersBiases.Count; i++)
            {
                _neuralNetwork.SetLayerBiasActivation(i, _layersBiases[i]);
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

        private double[] GetWeightsFromCurrentLine()
        {
            var weights = Array.ConvertAll(_currentLine.Split(TextSeparator), double.Parse);
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

        private void ThrowErrorFileLoadFile(string errorMsg)
        {
            throw new FileLoadException(ExceptionPostfix + errorMsg);
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
                case ActivationFunctionsNames.sinus:
                    return new ActivationSIN();
                case ActivationFunctionsNames.off:
                    return null;
                default:
                    throw new NotImplementedException();
            }
        }
    }
}
