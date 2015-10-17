namespace sieci_neuronowe
{
    using Encog;
    using System.IO;

    public class StreamStatusReportable : IStatusReportable
    {
        #region Fields

        private readonly StreamWriter _streamWriter;

        #endregion

        #region Constructors and Destructors

        public StreamStatusReportable(StreamWriter streamWriter)
        {
            _streamWriter = streamWriter;
        }

        #endregion

        #region Public Methods and Operators

        public void Report(int total, int current, string message)
        {
            _streamWriter.WriteLine(current + (total != 0 ? "/" + total : "") + " : " + message);
        }

        #endregion
    }
}