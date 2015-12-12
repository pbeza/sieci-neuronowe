using Microsoft.Win32;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace gui
{
    using geo;

    using OsmSharp.Math.Geo;

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow
    {
        private const string DllFile = "engine.dll";
        private const string DefaultExtension = ".osm";
        private const string OpenFileDialogFilter = "Open Street Maps file (*.osm) | *osm";
        private const string DefaultOsmFilePath = "liechtenstein-latest.osm";
        private const float ZoomFactor = 0.01f;
        private string _selectedOpenStreetMapFile = DefaultOsmFilePath;
        private Point _start;
        private Point _origin;
        private GeoData _geoData;

        [DllImport(DllFile, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.BStr)]
        public static extern string test();

        public MainWindow()
        {
            InitializeComponent();
        }

        private void OpenMenuItem_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                DefaultExt = DefaultExtension,
                Filter = OpenFileDialogFilter
            };
            var result = dlg.ShowDialog();
            if (result != true) return;
            _selectedOpenStreetMapFile = dlg.FileName;
            StartAlgorithm();
        }

        private void UIElement_OnMouseWheel(object sender, MouseWheelEventArgs e)
        {
            var st = (ScaleTransform)((TransformGroup)GeoImage.RenderTransform).Children.First(tr => tr is ScaleTransform);
            var zoom = e.Delta > 0 ? ZoomFactor : -ZoomFactor;
            st.ScaleX += zoom;
            st.ScaleY += zoom;
        }

        private void GeoImage_OnMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            GeoImage.CaptureMouse();
            var tt = (TranslateTransform)((TransformGroup)GeoImage.RenderTransform).Children.First(tr => tr is TranslateTransform);
            _start = e.GetPosition(GeoBorder);
            _origin = new Point(tt.X, tt.Y);
        }

        private void GeoImage_OnMouseMove(object sender, MouseEventArgs e)
        {
            if (!GeoImage.IsMouseCaptured) return;
            var tt = (TranslateTransform)((TransformGroup)GeoImage.RenderTransform).Children.First(tr => tr is TranslateTransform);
            var v = _start - e.GetPosition(GeoBorder);
            tt.X = _origin.X - v.X;
            tt.Y = _origin.Y - v.Y;
        }

        private void GeoImage_OnMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            GeoImage.ReleaseMouseCapture();
        }

        private void StartAlgorithm_Click(object sender, RoutedEventArgs e)
        {
            StartAlgorithm();
        }

        private void StartAlgorithm()
        {
            StatusBarText.Text = string.Format("File '{0}' loaded successfully. Starting algorithm...", _selectedOpenStreetMapFile);

            _geoData = new GeoData(_selectedOpenStreetMapFile);

            double[] lowerLeftPoint = { 47.0, 9.5 };
            double[] upperRightPoint = { 47.3, 9.7 };
            var corner1 = new GeoCoordinate(lowerLeftPoint[0], lowerLeftPoint[1]);
            var corner2 = new GeoCoordinate(upperRightPoint[0], upperRightPoint[1]);

            const int resolutionX = 1024;
            const int resolutionY = 1024;
            var temp = _geoData.GetTypesInArea(new GeoCoordinateBox(corner1, corner2), resolutionX, resolutionY);
            GeoTypeMap.Source = BitmapHelper.FromTypeArray(temp);

            StatusBarText.Text = "Algorithm ended successfully.";
        }
    }
}
