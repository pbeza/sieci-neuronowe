using Encog;
using System.IO;

namespace sieci_neuronowe
{
    public class StreamStatusReportable : IStatusReportable
    {
        private readonly StreamWriter streamWriter;

        public StreamStatusReportable(StreamWriter streamWriter)
        {
            this.streamWriter = streamWriter;
        }

        public void Report(int total, int current, string message)
        {
            this.streamWriter.WriteLine("{0} / {1} : {2}", current, total != 0 ? "/" + total : "", message);
        }
    }
}