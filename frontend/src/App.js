import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Upload, Download, Settings, Image, Zap, BarChart3, Clock, FileImage, Trash2, Play, Eye, Grid, Target } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const ContentAwareCompressionApp = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [compressionSettings, setCompressionSettings] = useState({
    qualityHigh: 90,
    qualityMedium: 60,
    qualityLow: 30,
    nSegments: 100
  });
  const [compressionTask, setCompressionTask] = useState(null);
  const [compressionResult, setCompressionResult] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isCompressing, setIsCompressing] = useState(false);
  const [activeVisualizationTab, setActiveVisualizationTab] = useState('original');
  const [imageLoadErrors, setImageLoadErrors] = useState({});
  const [compressionStarted, setCompressionStarted] = useState(false);
  const fileInputRef = useRef(null);
  const pollIntervalRef = useRef(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const handleFileSelect = useCallback((event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setUploadedFile(null);
      setCompressionResult(null);
      setCompressionTask(null);
      setCompressionStarted(false);
      setActiveVisualizationTab('original');
      setImageLoadErrors({});
    } else {
      alert('Please select a valid image file');
    }
  }, []);

  const uploadFile = useCallback(async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setUploadedFile(result);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('Upload failed: ' + error.message);
    } finally {
      setIsUploading(false);
    }
  }, [selectedFile]);

  const startCompression = useCallback(async () => {
    if (!uploadedFile) return;

    setIsCompressing(true);
    setCompressionStarted(true);
    setCompressionTask(null);
    setCompressionResult(null);
    setImageLoadErrors({});

    try {
      const formData = new FormData();
      formData.append('file_id', uploadedFile.file_id);
      formData.append('quality_high', compressionSettings.qualityHigh.toString());
      formData.append('quality_medium', compressionSettings.qualityMedium.toString());
      formData.append('quality_low', compressionSettings.qualityLow.toString());
      formData.append('n_segments', compressionSettings.nSegments.toString());

      const response = await fetch(`${API_BASE_URL}/compress`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setCompressionTask(result);
        startPolling(result.task_id);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Compression failed to start');
      }
    } catch (error) {
      console.error('Compression error:', error);
      alert('Compression failed: ' + error.message);
      setIsCompressing(false);
      setCompressionStarted(false);
    }
  }, [uploadedFile, compressionSettings]);

  const startPolling = useCallback((taskId) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }

    pollIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/status/${taskId}`);
        if (response.ok) {
          const status = await response.json();
          setCompressionTask(prev => ({ ...prev, ...status }));

          if (status.status === 'completed') {
            setCompressionResult(status.result);
            setIsCompressing(false);
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          } else if (status.status === 'failed') {
            alert('Compression failed: ' + (status.message || 'Unknown error'));
            setIsCompressing(false);
            setCompressionStarted(false);
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 1000);
  }, []);

  const downloadCompressed = useCallback(() => {
    if (compressionResult && compressionResult.compressed_path) {
      const fullPath = compressionResult.compressed_path;
      const filename = fullPath.split('/').pop();
      
      const link = document.createElement('a');
      link.href = `${API_BASE_URL}/download/${filename}`;
      link.download = filename;
      link.style.display = 'none';
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }, [compressionResult]);

  const cleanupFiles = useCallback(async () => {
    if (uploadedFile) {
      try {
        await fetch(`${API_BASE_URL}/cleanup/${uploadedFile.file_id}`, {
          method: 'DELETE',
        });
        setSelectedFile(null);
        setUploadedFile(null);
        setCompressionResult(null);
        setCompressionTask(null);
        setCompressionStarted(false);
        setImageLoadErrors({});
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      } catch (error) {
        console.error('Cleanup failed:', error);
      }
    }
  }, [uploadedFile]);

  const handleImageError = useCallback((vizType) => {
    console.error(`Failed to load ${vizType} visualization`);
    setImageLoadErrors(prev => ({
      ...prev,
      [vizType]: true
    }));
  }, []);

  const handleImageLoad = useCallback((vizType) => {
    console.log(`Successfully loaded ${vizType} visualization`);
    setImageLoadErrors(prev => ({
      ...prev,
      [vizType]: false
    }));
  }, []);

  const retryImageLoad = useCallback((vizType) => {
    setImageLoadErrors(prev => ({
      ...prev,
      [vizType]: false
    }));
    
    // Force image reload by adding timestamp
    const img = document.querySelector(`img[data-viz-type="${vizType}"]`);
    if (img) {
      const originalSrc = img.src.split('?')[0];
      img.src = `${originalSrc}?t=${Date.now()}`;
    }
  }, []);

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatTime = (seconds) => {
    return `${seconds.toFixed(2)}s`;
  };

  const renderVisualizationContent = (vizType) => {
    const isError = imageLoadErrors[vizType];
    
    const vizConfig = {
      original: {
        src: `${API_BASE_URL}/visualization/${uploadedFile?.file_id}/original?t=${Date.now()}`,
        alt: 'Original Image',
        title: `Original Image (${uploadedFile?.dimensions?.width} × ${uploadedFile?.dimensions?.height})`,
        description: 'Original uploaded image',
        icon: Image,
        fallbackText: 'Original image not available'
      },
      heatmap: {
        src: `${API_BASE_URL}/visualization/${uploadedFile?.file_id}/heatmap?t=${Date.now()}`,
        alt: 'Importance Heatmap',
        title: 'Importance Heatmap',
        description: 'Red areas indicate high importance regions that will be preserved with higher quality',
        icon: Target,
        fallbackText: 'Heatmap visualization not available'
      },
      segments: {
        src: `${API_BASE_URL}/visualization/${uploadedFile?.file_id}/segments?t=${Date.now()}`,
        alt: 'Image Segmentation',
        title: 'SLIC Segmentation',
        description: 'Different colors represent different segments and their importance levels',
        icon: Grid,
        fallbackText: 'Segmentation visualization not available'
      },
      compressed: {
        src: `${API_BASE_URL}/outputs/${compressionResult?.compressed_path}?t=${Date.now()}`,
        alt: 'Compressed Result',
        title: 'Compressed Result',
        description: 'Final compressed image with content-aware optimization',
        icon: Download,
        fallbackText: 'Compressed image not available'
      }
    };

    const config = vizConfig[vizType];
    if (!config) {
      return (
        <div className="text-center text-red-400">
          <p>Invalid visualization type: {vizType}</p>
        </div>
      );
    }

    const IconComponent = config.icon;

    return (
      <div className="text-center">
        {!isError ? (
          <div className="relative">
            <img
              src={config.src}
              alt={config.alt}
              data-viz-type={vizType}
              className="max-w-full max-h-96 rounded-lg shadow-lg mx-auto"
              onError={() => handleImageError(vizType)}
              onLoad={() => handleImageLoad(vizType)}
            />
          </div>
        ) : (
          <div className="text-gray-400 flex flex-col items-center justify-center min-h-[200px] bg-gray-800/30 rounded-lg border border-gray-600/30">
            <IconComponent className="w-16 h-16 mx-auto mb-4 text-gray-500" />
            <p className="text-lg font-medium">{config.fallbackText}</p>
            <p className="text-sm mt-2 text-gray-400">
              {vizType === 'compressed' && !compressionResult 
                ? 'Complete compression to view result'
                : vizType === 'segments' || vizType === 'heatmap'
                ? compressionStarted 
                  ? 'Visualization is being generated...'
                  : 'Start compression to generate visualization'
                : 'The visualization may still be processing or an error occurred'
              }
            </p>
            
            {/* Retry button - show for different conditions */}
            {((vizType === 'original') || 
              (vizType === 'compressed' && compressionResult) || 
              ((vizType === 'segments' || vizType === 'heatmap') && compressionStarted)) && (
              <button
                onClick={() => retryImageLoad(vizType)}
                className="mt-3 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
              >
                Retry Loading
              </button>
            )}
          </div>
        )}
        
        <div className="mt-4">
          <h3 className="text-lg font-semibold text-white">{config.title}</h3>
          <p className="text-sm text-gray-300 mt-2">{config.description}</p>
        </div>
      </div>
    );
  };

  const renderVisualizationTabs = () => {
    // Show visualization section as soon as file is uploaded
    if (!uploadedFile) return null;

    const tabs = [
      { 
        id: 'original', 
        label: 'Original Image', 
        icon: Image, 
        available: true,
        disabled: false
      },
      { 
        id: 'heatmap', 
        label: 'Importance Heatmap', 
        icon: Target, 
        available: compressionStarted,
        disabled: !compressionStarted
      },
      { 
        id: 'segments', 
        label: 'Segmentation', 
        icon: Grid, 
        available: compressionStarted,
        disabled: !compressionStarted
      },
      { 
        id: 'compressed', 
        label: 'Compressed Result', 
        icon: Download, 
        available: compressionResult,
        disabled: !compressionResult
      }
    ];

    return (
      <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 mb-8 border border-white/20">
        <h2 className="text-2xl font-bold mb-6 flex items-center">
          <Eye className="w-6 h-6 mr-2" />
          Image Analysis Visualization
        </h2>
        
        {/* Tab Navigation */}
        <div className="flex flex-wrap gap-2 mb-6">
          {tabs.map((tab) => {
            const IconComponent = tab.icon;
            const isActive = activeVisualizationTab === tab.id;
            const isDisabled = tab.disabled;
            
            return (
              <button
                key={tab.id}
                onClick={() => !isDisabled && setActiveVisualizationTab(tab.id)}
                disabled={isDisabled}
                className={`flex items-center px-4 py-2 rounded-lg transition-all duration-200 ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-lg'
                    : isDisabled
                    ? 'bg-gray-700/50 text-gray-500 cursor-not-allowed'
                    : 'bg-white/10 text-gray-300 hover:bg-white/20 hover:text-white cursor-pointer'
                }`}
              >
                <IconComponent className="w-4 h-4 mr-2" />
                {tab.label}
                {isDisabled && (
                  <span className="ml-2 text-xs opacity-60">
                    {tab.id === 'compressed' ? '(after compression)' : '(start compression)'}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        <div className="bg-white/5 rounded-xl p-6 min-h-[400px] flex items-center justify-center">
          {renderVisualizationContent(activeVisualizationTab)}
        </div>

        {/* Segment Statistics - only show when compression is complete */}
        {compressionResult && compressionResult.compression_stats && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-4 text-center">Segment Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-red-900/30 border border-red-500/30 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-red-400 mb-1">
                  {compressionResult.compression_stats.high || 0}
                </div>
                <div className="text-sm text-gray-300 mb-1">High Importance</div>
                <div className="text-xs text-gray-400">Quality: {compressionSettings.qualityHigh}%</div>
              </div>
              <div className="bg-yellow-900/30 border border-yellow-500/30 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-yellow-400 mb-1">
                  {compressionResult.compression_stats.medium || 0}
                </div>
                <div className="text-sm text-gray-300 mb-1">Medium Importance</div>
                <div className="text-xs text-gray-400">Quality: {compressionSettings.qualityMedium}%</div>
              </div>
              <div className="bg-green-900/30 border border-green-500/30 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-green-400 mb-1">
                  {compressionResult.compression_stats.low || 0}
                </div>
                <div className="text-sm text-gray-300 mb-1">Low Importance</div>
                <div className="text-xs text-gray-400">Quality: {compressionSettings.qualityLow}%</div>
              </div>
            </div>
          </div>
        )}

        {/* Processing indicator for visualizations */}
        {isCompressing && !compressionResult && (
          <div className="mt-6 bg-blue-900/30 border border-blue-500/30 rounded-lg p-4">
            <div className="flex items-center justify-center">
              <div className="animate-spin w-4 h-4 border-2 border-blue-400/30 border-t-blue-400 rounded-full mr-3"></div>
              <span className="text-blue-400">Generating visualizations during compression...</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Zap className="w-12 h-12 text-yellow-400 mr-4" />
            <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Smart Compress
            </h1>
          </div>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Advanced content-aware image compression using deep learning and SLIC segmentation
          </p>
        </div>

        <div className="max-w-6xl mx-auto">
          {/* File Upload Section */}
          <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 mb-8 border border-white/20">
            <h2 className="text-2xl font-bold mb-6 flex items-center">
              <Upload className="w-6 h-6 mr-2" />
              Upload Image
            </h2>
            
            <div className="space-y-6">
              <div
                className="border-2 border-dashed border-white/30 rounded-xl p-12 text-center hover:border-white/50 transition-colors cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <Image className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                <p className="text-lg mb-2">
                  {selectedFile ? selectedFile.name : 'Click to select an image or drag and drop'}
                </p>
                <p className="text-sm text-gray-400">
                  Supports JPG, PNG, WebP formats
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>

              {selectedFile && !uploadedFile && (
                <div className="flex items-center justify-between bg-white/5 rounded-lg p-4">
                  <div className="flex items-center">
                    <FileImage className="w-6 h-6 mr-3 text-blue-400" />
                    <div>
                      <p className="font-medium">{selectedFile.name}</p>
                      <p className="text-sm text-gray-400">{formatFileSize(selectedFile.size)}</p>
                    </div>
                  </div>
                  <button
                    onClick={uploadFile}
                    disabled={isUploading}
                    className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-6 py-2 rounded-lg transition-colors flex items-center"
                  >
                    {isUploading ? (
                      <>
                        <div className="animate-spin w-4 h-4 border-2 border-white/30 border-t-white rounded-full mr-2"></div>
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload className="w-4 h-4 mr-2" />
                        Upload
                      </>
                    )}
                  </button>
                </div>
              )}

              {uploadedFile && (
                <div className="bg-green-900/30 border border-green-500/30 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-green-400 rounded-full mr-3"></div>
                      <div>
                        <p className="font-medium">Upload Complete</p>
                        <p className="text-sm text-gray-400">
                          {uploadedFile.dimensions?.width} × {uploadedFile.dimensions?.height} pixels • {formatFileSize(uploadedFile.size)}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={cleanupFiles}
                      className="text-red-400 hover:text-red-300 p-2 rounded transition-colors"
                      title="Remove files and start over"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Compression Settings */}
          {uploadedFile && (
            <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 mb-8 border border-white/20">
              <h2 className="text-2xl font-bold mb-6 flex items-center">
                <Settings className="w-6 h-6 mr-2" />
                Compression Settings
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    High Importance Quality: <span className="text-red-400 font-mono">{compressionSettings.qualityHigh}%</span>
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="100"
                    value={compressionSettings.qualityHigh}
                    onChange={(e) => setCompressionSettings(prev => ({
                      ...prev,
                      qualityHigh: parseInt(e.target.value)
                    }))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <div className="flex justify-between text-xs text-gray-400 mt-1">
                    <span>50%</span>
                    <span>100%</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Medium Importance Quality: <span className="text-yellow-400 font-mono">{compressionSettings.qualityMedium}%</span>
                  </label>
                  <input
                    type="range"
                    min="30"
                    max="90"
                    value={compressionSettings.qualityMedium}
                    onChange={(e) => setCompressionSettings(prev => ({
                      ...prev,
                      qualityMedium: parseInt(e.target.value)
                    }))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <div className="flex justify-between text-xs text-gray-400 mt-1">
                    <span>30%</span>
                    <span>90%</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Low Importance Quality: <span className="text-green-400 font-mono">{compressionSettings.qualityLow}%</span>
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="70"
                    value={compressionSettings.qualityLow}
                    onChange={(e) => setCompressionSettings(prev => ({
                      ...prev,
                      qualityLow: parseInt(e.target.value)
                    }))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <div className="flex justify-between text-xs text-gray-400 mt-1">
                    <span>10%</span>
                    <span>70%</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Number of Segments: <span className="text-blue-400 font-mono">{compressionSettings.nSegments}</span>
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="500"
                    value={compressionSettings.nSegments}
                    onChange={(e) => setCompressionSettings(prev => ({
                      ...prev,
                      nSegments: parseInt(e.target.value)
                    }))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <div className="flex justify-between text-xs text-gray-400 mt-1">
                    <span>50</span>
                    <span>500</span>
                  </div>
                </div>
              </div>

              <div className="mt-8">
                <button
                  onClick={startCompression}
                  disabled={isCompressing}
                  className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed px-8 py-3 rounded-lg transition-all duration-200 flex items-center text-lg font-medium shadow-lg hover:shadow-xl"
                >
                  {isCompressing ? (
                    <>
                      <div className="animate-spin w-5 h-5 border-2 border-white/30 border-t-white rounded-full mr-3"></div>
                      Processing...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5 mr-3" />
                      Start Compression
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Compression Progress */}
          {compressionTask && isCompressing && (
            <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 mb-8 border border-white/20">
              <h2 className="text-2xl font-bold mb-6 flex items-center">
                <Clock className="w-6 h-6 mr-2" />
                Processing Progress
              </h2>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-lg capitalize">Status: {compressionTask.status}</span>
                  <div className="flex items-center">
                    <div className="animate-spin w-4 h-4 border-2 border-blue-400/30 border-t-blue-400 rounded-full mr-2"></div>
                    <span className="text-blue-400">Processing</span>
                  </div>
                </div>
                
                {compressionTask.progress !== undefined && (
                  <div>
                    <div className="flex justify-between mb-2">
                      <span>Progress</span>
                      <span className="font-mono">{Math.round(compressionTask.progress)}%</span>
                    </div>
                    <div className="w-full bg-white/20 rounded-full h-3 overflow-hidden">
                      <div 
                        className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300 ease-out"
                        style={{ width: `${Math.max(0, Math.min(100, compressionTask.progress))}%` }}
                      ></div>
                    </div>
                  </div>
                )}

                {compressionTask.message && (
                  <div className="bg-white/5 rounded-lg p-3">
                    <p className="text-gray-300 text-sm">{compressionTask.message}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Visualization Section */}
          {renderVisualizationTabs()}

          {/* Compression Results */}
          {compressionResult && (
            <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 mb-8 border border-white/20">
              <h2 className="text-2xl font-bold mb-6 flex items-center">
                <BarChart3 className="w-6 h-6 mr-2" />
                Compression Results
              </h2>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
                <div className="bg-white/5 rounded-xl p-6 text-center">
                  <div className="text-3xl font-bold text-green-400 mb-2">
                    {compressionResult.reduction_percentage ? 
                      `${compressionResult.reduction_percentage.toFixed(1)}%` : 'N/A'}
                  </div>
                  <div className="text-sm text-gray-300">Size Reduction</div>
                </div>
                
                <div className="bg-white/5 rounded-xl p-6 text-center">
                  <div className="text-3xl font-bold text-blue-400 mb-2">
                    {compressionResult.original_size ? 
                      formatFileSize(compressionResult.original_size) : 'N/A'}
                  </div>
                  <div className="text-sm text-gray-300">Original Size</div>
                </div>
                
                <div className="bg-white/5 rounded-xl p-6 text-center">
                  <div className="text-3xl font-bold text-purple-400 mb-2">
                    {compressionResult.compressed_size ? 
                      formatFileSize(compressionResult.compressed_size) : 'N/A'}
                  </div>
                  <div className="text-sm text-gray-300">Compressed Size</div>
                </div>
                
                <div className="bg-white/5 rounded-xl p-6 text-center">
                  <div className="text-3xl font-bold text-yellow-400 mb-2">
                    {compressionResult.processing_time ? 
                      formatTime(compressionResult.processing_time) : 'N/A'}
                  </div>
                  <div className="text-sm text-gray-300">Processing Time</div>
                </div>
              </div>

              {compressionResult.space_saved && (
                <div className="bg-green-900/30 border border-green-500/30 rounded-lg p-4 mb-6">
                  <p className="text-center text-lg">
                    <span className="text-green-400 font-bold text-xl">
                      {formatFileSize(compressionResult.space_saved)}
                    </span>
                    <span className="text-gray-300"> saved • </span>
                    <span className="text-green-300 font-semibold">
                      {((compressionResult.space_saved / compressionResult.original_size) * 100).toFixed(1)}% reduction
                    </span>
                  </p>
                </div>
              )}

              <div className="flex justify-center space-x-4">
                <button
                  onClick={downloadCompressed}
                  className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 px-8 py-3 rounded-lg transition-all duration-200 flex items-center text-lg font-medium shadow-lg hover:shadow-xl"
                >
                  <Download className="w-5 h-5 mr-3" />
                  Download Compressed Image
                </button>
              </div>
            </div>
          )}

          {/* Info Section */}
          <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-8 border border-white/10">
            <h3 className="text-xl font-bold mb-4">How Smart Compression Works</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-gray-300">
              <div className="text-center">
                <Grid className="w-12 h-12 mx-auto mb-3 text-blue-400" />
                <h4 className="font-medium text-white mb-2">1. SLIC Segmentation</h4>
                <p>The image is intelligently divided into superpixels using Simple Linear Iterative Clustering for content-aware analysis.</p>
              </div>
              <div className="text-center">
                <Target className="w-12 h-12 mx-auto mb-3 text-purple-400" />
                <h4 className="font-medium text-white mb-2">2. Importance Analysis</h4>
                <p>Deep learning models analyze each segment to determine visual importance - faces, text, and objects get priority.</p>
              </div>
              <div className="text-center">
                <Zap className="w-12 h-12 mx-auto mb-3 text-yellow-400" />
                <h4 className="font-medium text-white mb-2">3. Adaptive Compression</h4>
                <p>Different compression levels are applied based on importance - preserving quality where it matters most while maximizing size reduction.</p>
              </div>
            </div>
            
            <div className="mt-8 pt-6 border-t border-white/10">
              <h4 className="font-medium text-white mb-3">Compression Quality Levels:</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-red-400 rounded-full mr-2"></div>
                  <span><strong>High:</strong> Faces, text, detailed objects</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-yellow-400 rounded-full mr-2"></div>
                  <span><strong>Medium:</strong> Moderate detail regions</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-green-400 rounded-full mr-2"></div>
                  <span><strong>Low:</strong> Backgrounds, sky, uniform areas</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid #1e40af;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid #1e40af;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .slider::-webkit-slider-track {
          height: 8px;
          border-radius: 4px;
          background: rgba(255,255,255,0.2);
        }
        
        .slider::-moz-range-track {
          height: 8px;
          border-radius: 4px;
          background: rgba(255,255,255,0.2);
        }
      `}</style>
    </div>
  );
};

export default ContentAwareCompressionApp;