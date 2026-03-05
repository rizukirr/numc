#ifndef NUMC_EXPORT_H
#define NUMC_EXPORT_H

#if defined(NUMC_STATIC_DEFINE)
#define NUMC_API
#else
#ifndef NUMC_API
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef numc_EXPORTS
#define NUMC_API __declspec(dllexport)
#else
#define NUMC_API __declspec(dllimport)
#endif
#else
#if __GNUC__ >= 4
#define NUMC_API __attribute__((visibility("default")))
#else
#define NUMC_API
#endif
#endif
#endif
#endif

#endif /* NUMC_EXPORT_H */
