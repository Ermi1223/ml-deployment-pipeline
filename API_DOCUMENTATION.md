# TensorFlow Serving API Documentation

## Overview

This document provides comprehensive documentation for the TensorFlow Serving APIs used in the MNIST classification pipeline. The system supports both REST and gRPC APIs for model inference and management.

## Base URLs

- **REST API**: `http://localhost:8501/v1/models/mnist`
- **gRPC API**: `localhost:8500`

## Authentication

All API endpoints require authentication using API keys:

```http
Authorization: Bearer <API_KEY>