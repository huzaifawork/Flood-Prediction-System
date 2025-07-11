import { motion } from 'framer-motion';
import { FloodRiskLevel, getRiskColor } from '../types';

interface RiskBadgeProps {
  riskLevel: FloodRiskLevel;
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
  animate?: boolean;
}

const RiskBadge: React.FC<RiskBadgeProps> = ({ 
  riskLevel, 
  showLabel = true, 
  size = 'md',
  animate = false 
}) => {
  // Determine size classes
  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-2'
  };
  
  // Get color class based on risk level
  const colorClass = getRiskColor(riskLevel);
  
  // Animation variants
  const variants = {
    initial: { scale: 0.8, opacity: 0 },
    animate: { scale: 1, opacity: 1 },
    pulse: {
      scale: [1, 1.05, 1],
      transition: { 
        repeat: Infinity, 
        repeatType: "reverse", 
        duration: 1.5 
      }
    }
  };

  return (
    <motion.span
      className={`risk-badge ${colorClass} ${sizeClasses[size]}`}
      initial={animate ? "initial" : undefined}
      animate={animate ? ["animate", "pulse"] : undefined}
      variants={variants}
    >
      {showLabel ? riskLevel : ''}
    </motion.span>
  );
};

export default RiskBadge; 