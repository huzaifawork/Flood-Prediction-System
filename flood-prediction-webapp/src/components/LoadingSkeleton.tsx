import { motion } from 'framer-motion';

interface LoadingSkeletonProps {
  variant?: 'card' | 'weather' | 'chart' | 'list' | 'text';
  className?: string;
  count?: number;
}

const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({ 
  variant = 'card', 
  className = '',
  count = 1 
}) => {
  const shimmer = {
    initial: { backgroundPosition: '-200px 0' },
    animate: { backgroundPosition: 'calc(200px + 100%) 0' },
    transition: {
      duration: 1.5,
      ease: 'linear',
      repeat: Infinity,
    },
  };

  const skeletonBase = "bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200 dark:from-gray-700 dark:via-gray-600 dark:to-gray-700 rounded animate-pulse";

  const renderWeatherSkeleton = () => (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className={`h-5 w-32 ${skeletonBase} mb-2`} />
          <div className={`h-3 w-16 ${skeletonBase}`} />
        </div>
        <div className={`h-8 w-8 rounded-full ${skeletonBase}`} />
      </div>

      {/* Temperature */}
      <div className="text-center mb-6">
        <div className={`h-12 w-24 ${skeletonBase} mx-auto mb-2`} />
        <div className={`h-4 w-40 ${skeletonBase} mx-auto mb-2`} />
        <div className={`h-3 w-32 ${skeletonBase} mx-auto`} />
      </div>

      {/* Details Grid */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="flex items-center space-x-2 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <div className={`h-4 w-4 rounded ${skeletonBase}`} />
            <div>
              <div className={`h-3 w-12 ${skeletonBase} mb-1`} />
              <div className={`h-4 w-16 ${skeletonBase}`} />
            </div>
          </div>
        ))}
      </div>

      {/* Button */}
      <div className={`h-12 w-full ${skeletonBase}`} />
    </div>
  );

  const renderChartSkeleton = () => (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
      <div className={`h-6 w-48 ${skeletonBase} mb-6`} />
      <div className="space-y-4">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="flex items-end space-x-2">
            <div className={`h-${8 + (i % 3) * 4} w-8 ${skeletonBase}`} />
            <div className={`h-${6 + (i % 4) * 3} w-8 ${skeletonBase}`} />
            <div className={`h-${10 + (i % 2) * 6} w-8 ${skeletonBase}`} />
            <div className={`h-${4 + (i % 5) * 2} w-8 ${skeletonBase}`} />
          </div>
        ))}
      </div>
    </div>
  );

  const renderListSkeleton = () => (
    <div className="space-y-3">
      {[...Array(count)].map((_, i) => (
        <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
          <div className="flex items-center space-x-3">
            <div className={`h-5 w-5 rounded ${skeletonBase}`} />
            <div>
              <div className={`h-4 w-24 ${skeletonBase} mb-1`} />
              <div className={`h-3 w-16 ${skeletonBase}`} />
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`h-3 w-8 ${skeletonBase}`} />
            <div className={`h-4 w-12 ${skeletonBase}`} />
          </div>
        </div>
      ))}
    </div>
  );

  const renderCardSkeleton = () => (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
      <div className={`h-6 w-3/4 ${skeletonBase} mb-4`} />
      <div className={`h-4 w-full ${skeletonBase} mb-2`} />
      <div className={`h-4 w-5/6 ${skeletonBase} mb-4`} />
      <div className="flex space-x-4">
        <div className={`h-10 w-24 ${skeletonBase}`} />
        <div className={`h-10 w-24 ${skeletonBase}`} />
      </div>
    </div>
  );

  const renderTextSkeleton = () => (
    <div className="space-y-2">
      {[...Array(count)].map((_, i) => (
        <div key={i} className={`h-4 ${i === count - 1 ? 'w-3/4' : 'w-full'} ${skeletonBase}`} />
      ))}
    </div>
  );

  const renderSkeleton = () => {
    switch (variant) {
      case 'weather':
        return renderWeatherSkeleton();
      case 'chart':
        return renderChartSkeleton();
      case 'list':
        return renderListSkeleton();
      case 'text':
        return renderTextSkeleton();
      default:
        return renderCardSkeleton();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className={className}
    >
      {renderSkeleton()}
    </motion.div>
  );
};

export default LoadingSkeleton;
