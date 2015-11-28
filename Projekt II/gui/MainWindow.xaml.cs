using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;

namespace gui
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private const string DefaultExtension = ".png";
        private const string OpenFileDialogFilter = "Image files (*.jpg, *.jpeg, *.jpe, *.jfif, *.png) | *.jpg; *.jpeg; *.jpe; *.jfif; *.png";
        private const float ZoomFactor = 0.01f;
        private BitmapImage _geoImg;
        private Point _start;
        private Point _origin;

        [DllImport("engine.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.BStr)]
        public static extern string test();

        public MainWindow()
        {
            InitializeComponent();
        }

        private void InitImage(string path)
        {
            _geoImg = new BitmapImage();
            _geoImg.BeginInit();
            _geoImg.UriSource = new Uri(path);
            _geoImg.EndInit();
            GeoImage.Source = _geoImg;
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
            InitImage(dlg.FileName);
            StatusBarText.Text = string.Format("File '{0}' loaded successfully.", dlg.FileName);
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
            StatusBarText.Text = "Wait. Starting algorithm...";
            // TODO
            Console.WriteLine(test());
            StatusBarText.Text = "Algorithm ended successfully.";
        }
    }
}
