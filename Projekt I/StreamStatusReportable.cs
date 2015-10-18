namespace sieci_neuronowe
{
    using Encog;
    using System.IO;

    public class StreamStatusReportable : IStatusReportable
    {
        private readonly StreamWriter _streamWriter;

        public StreamStatusReportable(StreamWriter streamWriter)
        {
            _streamWriter = streamWriter;
        }

        public void Report(int total, int current, string message)
        {
            _streamWriter.WriteLine(current + (total != 0 ? "/" + total : "") + " : " + message);
        }
    }
}