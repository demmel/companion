import React, { Component, ReactNode } from "react";
import { css } from "@styled-system/css";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);

    // Report error details
    const errorReport = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
    };

    console.error("Full error report:", errorReport);

    // You could send this to an error reporting service here
    // Example: errorReportingService.report(errorReport);

    this.setState({
      error,
      errorInfo,
    });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          className={css({
            p: 6,
            maxWidth: "800px",
            margin: "0 auto",
            backgroundColor: "red.50",
            border: "2px solid",
            borderColor: "red.200",
            borderRadius: "md",
            mt: 8,
          })}
        >
          <h2
            className={css({
              fontSize: "xl",
              fontWeight: "bold",
              color: "red.800",
              mb: 4,
            })}
          >
            Something went wrong
          </h2>

          <p
            className={css({
              color: "red.700",
              mb: 4,
            })}
          >
            The application encountered an error. The error details have been
            logged to the console.
          </p>

          <details
            className={css({
              backgroundColor: "white",
              border: "1px solid",
              borderColor: "red.300",
              borderRadius: "sm",
              p: 3,
              mb: 4,
            })}
          >
            <summary
              className={css({
                cursor: "pointer",
                fontWeight: "medium",
                color: "red.800",
              })}
            >
              Error Details
            </summary>

            <div
              className={css({
                mt: 2,
                fontSize: "sm",
                fontFamily: "mono",
                whiteSpace: "pre-wrap",
                color: "red.600",
              })}
            >
              <strong>Error:</strong> {this.state.error?.message}
              {this.state.error?.stack && (
                <div className={css({ mt: 2 })}>
                  <strong>Stack Trace:</strong>
                  <pre
                    className={css({
                      mt: 1,
                      p: 2,
                      backgroundColor: "red.100",
                      borderRadius: "sm",
                      overflow: "auto",
                      fontSize: "xs",
                    })}
                  >
                    {this.state.error.stack}
                  </pre>
                </div>
              )}
              {this.state.errorInfo?.componentStack && (
                <div className={css({ mt: 2 })}>
                  <strong>Component Stack:</strong>
                  <pre
                    className={css({
                      mt: 1,
                      p: 2,
                      backgroundColor: "red.100",
                      borderRadius: "sm",
                      overflow: "auto",
                      fontSize: "xs",
                    })}
                  >
                    {this.state.errorInfo.componentStack}
                  </pre>
                </div>
              )}
            </div>
          </details>

          <button
            onClick={() => window.location.reload()}
            className={css({
              backgroundColor: "red.600",
              color: "white",
              px: 4,
              py: 2,
              borderRadius: "md",
              border: "none",
              cursor: "pointer",
              fontWeight: "medium",
              _hover: {
                backgroundColor: "red.700",
              },
            })}
          >
            Reload Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
