/*
 * Copyright (c) 2008 - 2014 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

/**
 * This file was factored out of sutil.h because of a number of header files
 * that require SUTILAPI for proper exporting, but for which including all of
 * sutil.h introduces problems, not the least of which is header pollution from
 * the transitive inclusion of optix.h.
 */

#ifndef __samples_util_sutilapi_h__
#define __samples_util_sutilapi_h__

#ifndef SUTILAPI
#  if sutil_EXPORTS /* Set by CMAKE */
#    if defined( _WIN32 ) || defined( _WIN64 )
#      define SUTILAPI __declspec(dllexport) 
#      define SUTILCLASSAPI
#    elif defined( linux ) || defined( __linux ) || defined ( __CYGWIN__ )
#      define SUTILAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI SUTILAPI
#    elif defined( __APPLE__ ) && defined( __MACH__ )
#      define SUTILAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI SUTILAPI
#    else
#      error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#    endif

#  else /* sutil_EXPORTS */

#    if defined( _WIN32 ) || defined( _WIN64 )
#      define SUTILAPI __declspec(dllimport)
#      define SUTILCLASSAPI
#    elif defined( linux ) || defined( __linux ) || defined ( __CYGWIN__ )
#      define SUTILAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI SUTILAPI
#    elif defined( __APPLE__ ) && defined( __MACH__ )
#      define SUTILAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI SUTILAPI
#    else
#      error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#    endif

#  endif /* sutil_EXPORTS */
#endif

#endif /* __samples_util_sutilapi_h__ */
