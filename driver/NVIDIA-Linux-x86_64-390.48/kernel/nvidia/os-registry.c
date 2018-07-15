/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2000-2016 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#define  __NO_VERSION__
#define NV_DEFINE_REGISTRY_KEY_TABLE
#include "nv-misc.h"
#include "os-interface.h"
#include "nv-linux.h"
#include "nv-reg.h"
#include "nv-gpu-info.h"

/*!
 * @brief This function parses the PCI BDF identifier string and returns the 
 * Domain, Bus, Device and function components from the PCI BDF string.
 * 
 * This parser is highly adaptable and hence allows PCI BDF string in following 
 * 3 formats.
 *
 * 1)  bus:slot                 : Domain and function defaults to 0.
 * 2)  domain:bus:slot          : Function defaults to 0.
 * 3)  domain:bus:slot.func     : Complete PCI dev id string.
 *
 * This parser is shared between 2 module parameters, namely: 
 * NVreg_RegistryDwordsPerDevice and NVreg_AssignGpus.
 *
 * @param[in]  pci_dev_str      String containing the BDF to be parsed. 
 * @param[out] pci_domain       Pointer where pci_domain is to be returned.
 * @param[out] pci_bus          Pointer where pci_bus is to be returned.
 * @param[out] pci_slot         Pointer where pci_slot is to be returned.
 * @param[out] pci_func         Pointer where pci_func is to be returned.
 *
 * @return NV_TRUE if succeeds, or NV_FALSE otherwise.
 */
static NV_STATUS pci_str_to_bdf(char *pci_dev_str, NvU32 *pci_domain, 
    NvU32 *pci_bus, NvU32 *pci_slot, NvU32 *pci_func)
{
    char *option_string = NULL;
    char *token, *string;
    NvU32 domain, bus, slot;
    NV_STATUS status = NV_OK;

    //
    // remove_spaces() allocates memory, hence we need to keep a pointer
    // to the original string for freeing at end of function.
    //
    if ((option_string = rm_remove_spaces(pci_dev_str)) == NULL)
    {
        // memory allocation failed, returning
        return NV_ERR_GENERIC;
    }

    string = option_string;

    if (!strlen(string) || !pci_domain || !pci_bus || !pci_slot || !pci_func)
    {
        status = NV_ERR_INVALID_ARGUMENT;
        goto done;
    }

    if ((token = strsep(&string, ".")) != NULL)
    {
        // PCI device can have maximum 8 functions only.
        if ((string != NULL) && (!(*string >= '0' && *string <= '7') ||
            (strlen(string) > 1)))
        {
            nv_printf(NV_DBG_ERRORS,
                      "NVRM: Invalid PCI function in token %s\n",
                      pci_dev_str);
            status = NV_ERR_INVALID_ARGUMENT;
            goto done;
        }
        else if (string == NULL)
        {
            *pci_func = 0;
        }
        else
        {
            *pci_func = (NvU32)(*string - '0');
        }

        domain = simple_strtoul(token, &string, 16);

        if ((string == NULL) || (*string != ':') || (*(string + 1) == '\0'))
        {
            nv_printf(NV_DBG_ERRORS,
                      "NVRM: Invalid PCI domain/bus in token %s\n",
                      pci_dev_str);
            status = NV_ERR_INVALID_ARGUMENT;
            goto done;
        }

        token = string;
        bus = simple_strtoul((token + 1), &string, 16);

        if (string == NULL)
        {
            nv_printf(NV_DBG_ERRORS,
                      "NVRM: Invalid PCI bus/slot in token %s\n",
                      pci_dev_str);
            status = NV_ERR_INVALID_ARGUMENT;
            goto done;
        }

        if (*string != '\0')
        {
            if ((*string != ':') || (*(string + 1) == '\0'))
            {
                nv_printf(NV_DBG_ERRORS,
                          "NVRM: Invalid PCI slot in token %s\n",
                          pci_dev_str);
                status = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }

            token = string;
            slot = (NvU32)simple_strtoul(token + 1, &string, 16);
            if ((slot == 0) && ((token + 1) == string))
            {
                nv_printf(NV_DBG_ERRORS,
                          "NVRM: Invalid PCI slot in token %s\n",
                          pci_dev_str);
                status = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }
            *pci_domain = domain;
            *pci_bus = bus;
            *pci_slot = slot;
        }
        else
        {
            *pci_slot = bus;
            *pci_bus = domain;
            *pci_domain = 0;
        }
        status = NV_OK;
    }
    else
    {
        status = NV_ERR_INVALID_ARGUMENT;
    }

done:
    // Freeing the memory allocated by remove_spaces().
    os_free_mem(option_string);
    return status;
}

/*!
 * @brief This function parses the registry keys per GPU device. It accepts a 
 * semicolon separated list of key=value pairs. The first key value pair MUST be
 * "pci=DDDD:BB:DD.F;" where DDDD is Domain, BB is Bus Id, DD is device slot
 * number and F is the Function. This PCI BDF is used to identify which GPU to
 * assign the registry keys that follows next.
 * If a GPU corresponding to the value specified in "pci=DDDD:BB:DD.F;" is NOT
 * found, then all the registry keys that follows are skipped, until we find next
 * valid pci identified "pci=DDDD:BB:DD.F;". Following are the valid formats for
 * the value of the "pci" string:
 * 1)  bus:slot                 : Domain and function defaults to 0.
 * 2)  domain:bus:slot          : Function defaults to 0.
 * 3)  domain:bus:slot.func     : Complete PCI dev id string.
 *
 *
 * @param[in]  sp       pointer to nvidia_stack_t struct.
 *
 * @return NV_OK if succeeds, or NV_STATUS error code otherwise.
 */
NV_STATUS nv_parse_per_device_option_string(nvidia_stack_t *sp)
{
    NV_STATUS status = NV_OK;
    char *option_string = NULL;
    char *ptr, *token;
    char *name, *value;
    NvU32 data, domain, bus, slot, func;
    nv_linux_state_t *nvl = NULL;
    nv_state_t *nv = NULL;

    if (NVreg_RegistryDwordsPerDevice != NULL)
    {
        if ((option_string = rm_remove_spaces(NVreg_RegistryDwordsPerDevice)) == NULL)
        {
            return NV_ERR_GENERIC;
        }

        ptr = option_string;

        while ((token = strsep(&ptr, ";")) != NULL)
        {
            if (!(name = strsep(&token, "=")) || !strlen(name))
            {
                continue;
            }

            if (!(value = strsep(&token, "=")) || !strlen(value))
            {
                continue;
            }

            if (strsep(&token, "=") != NULL)
            {
                continue;
            }

            // If this key is "pci", then value is pci_dev id string
            // which needs special parsing as it is NOT a dword.
            if (strcmp(name, NV_REG_PCI_DEVICE_BDF) == 0)
            {
                status = pci_str_to_bdf(value, &domain, &bus, &slot, &func);

                // Check if PCI_DEV id string was in a valid format or NOT.
                if (NV_OK != status)
                {
                    // lets reset cached pci dev
                    nv = NULL;
                }
                else
                {
                    nvl = find_pci(domain, bus, slot, func);
                    //
                    // If NO GPU found corresponding to this GPU, then reset
                    // cached state. This helps ignore the following registry
                    // keys until valid PCI BDF is found in the commandline.
                    //
                    if (!nvl)
                    {
                        nv = NULL;
                    }
                    else
                    {
                        nv = NV_STATE_PTR(nvl);
                    }
                }
                continue;
            }

            //
            // Check if cached pci_dev string in the commandline is in valid
            // format, else we will skip all the successive registry entries
            // (<key, value> pairs) until a valid PCI_DEV string is encountered
            // in the commandline.
            //
            if (!nv)
                continue;

            data = (NvU32)simple_strtoul(value, NULL, 0);

            rm_write_registry_dword(sp, nv, "NVreg", name, data);
        }

        os_free_mem(option_string);
    }
    return status;
}

static NvBool parse_assign_gpus_string(void)
{
    char *option_string = NULL;
    char *ptr, *token;
    NvU32 domain, bus, slot, func;

    if (NVreg_AssignGpus == NULL)
    {
        return NV_FALSE;
    }

    if ((option_string = rm_remove_spaces(NVreg_AssignGpus)) == NULL)
    {
        return NV_FALSE;
    }

    ptr = option_string;

    // token string should be in formats:
    //   bus:slot
    //   domain:bus:slot
    //   domain:bus:slot.func

    while ((token = strsep(&ptr, ",")) != NULL)
    {
        if (NV_OK != pci_str_to_bdf(token, &domain, &bus, &slot, &func))
        {
            nv_printf(NV_DBG_ERRORS,
                      "NVRM: Invalid PCI device id token %s\n",
                      token);
            continue;
        }

        // GPUs are always function 0.
        if (0 != func)
        {
            nv_printf(NV_DBG_ERRORS,
                      "NVRM: Invalid PCI function in token %s\n",
                      token);
            continue;
        }

        nv_assign_gpu_pci_info[nv_assign_gpu_count].domain = domain;
        nv_assign_gpu_pci_info[nv_assign_gpu_count].bus = bus;
        nv_assign_gpu_pci_info[nv_assign_gpu_count].slot = slot;
        nv_assign_gpu_count++;

        if (nv_assign_gpu_count == NV_MAX_DEVICES)
            break;
    }

    os_free_mem(option_string);

    return (nv_assign_gpu_count ? NV_TRUE : NV_FALSE);
}

static void test_and_modify_registry_value(char* name, NvU32 default_data, NvU32 data)
{
    unsigned int i;
    nv_parm_t* entry;
    for (i = 0; (entry = &nv_parms[i])->name != NULL; i++)
    {
        if (strcmp(entry->name, name) == 0)
        {
            if (*(entry->data) == default_data)
                *(entry->data) = data;
            return;
        }
    }
}

static void detect_virtualization_and_apply_defaults(nv_stack_t *sp)
{
#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
    if (nv_is_virtualized_system(sp))
    {
        test_and_modify_registry_value(NV_REG_STRING(__NV_CHECK_PCI_CONFIG_SPACE),
                                           NV_CHECK_PCI_CONFIG_SPACE_INIT,
                                           NV_CHECK_PCI_CONFIG_SPACE_DISABLED);
        return;
    }
#endif
    test_and_modify_registry_value(NV_REG_STRING(__NV_CHECK_PCI_CONFIG_SPACE),
                                       NV_CHECK_PCI_CONFIG_SPACE_INIT,
                                       NV_CHECK_PCI_CONFIG_SPACE_ENABLED);
}

NV_STATUS NV_API_CALL os_registry_init(void)
{
    nv_parm_t *entry;
    unsigned int i;
    nvidia_stack_t *sp = NULL;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    if (NVreg_RmMsg != NULL)
    {
        rm_write_registry_string(sp, NULL, "NVreg",
                "RmMsg", NVreg_RmMsg, strlen(NVreg_RmMsg));
    }

    memset(&nv_assign_gpu_pci_info, 0, sizeof(nv_assign_gpu_pci_info));

    if (parse_assign_gpus_string())
    {
        rm_write_registry_string(sp, NULL, "NVreg", NV_REG_ASSIGN_GPUS,
                                 NVreg_AssignGpus, strlen(NVreg_AssignGpus));
    }

    rm_parse_option_string(sp, NVreg_RegistryDwords);

    detect_virtualization_and_apply_defaults(sp);

    for (i = 0; (entry = &nv_parms[i])->name != NULL; i++)
    {
        rm_write_registry_dword(sp, NULL, entry->node, entry->name, *entry->data);
    }

    nv_kmem_cache_free_stack(sp);

    return NV_OK;
}
